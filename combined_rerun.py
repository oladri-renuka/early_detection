"""
Combined rerun — final GPU job addressing all reviewer-flagged issues.

Part A: GSM8K + MATH-500
  - Forced sweep at 32/64/128/256/512 tokens  (replaces budget_forcing_gsm_math.py)
  - Uncapped baseline on same 500 problems    (provides valid comparison denominator)
  - Answer cap raised to 1,000 tokens         (fixes 200-token truncation from item 3)
  - Same problem set for every cell           (fixes item 2: n mismatch)
  - B=512 added                               (fixes item 11: 256-512 unmeasured)

Part B: AIME fixed caps
  - Re-runs budget_forcing_fixed_cap.py       (fixes item 3: re-prompt was only 32 tokens)
  - Answer cap raised to 1,000 tokens
  - Uses same seed=0 problem set as original

Part C: Temperature rerun on correct AIME problems
  - Uses ds.shuffle(seed=42) — same 200 as records.json  (fixes item 1: alignment)
  - 4 seeds, T=0.6, majority vote
  - Hash-verified problem alignment
  - Reports independence baseline + Cohen's kappa

Part D: Two greedy runs on correct AIME problems
  - Uses ds.shuffle(seed=42) — same 200 as records.json  (fixes item 1)
  - Hash-verified problem alignment
  - Reports independence baseline + Cohen's kappa

Outputs:
  results/final_gsm_math.json       (Part A)
  results/final_aime_fixed_cap.json (Part B)
  results/final_temp_rerun.json     (Part C)
  results/final_greedy_rerun.json   (Part D)

Usage:
  python combined_rerun.py                    # all parts
  python combined_rerun.py --parts A B C D   # select parts
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

MODEL_NAME  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_TOKENS  = 10_000
ANSWER_CAP  = 1_000   # raised from 32/200/256 — fixes item 3
TEMPERATURE = 0.6
TEMP_SEEDS  = [0, 1, 2, 3]
GSM_BUDGETS = [32, 64, 128, 256, 512]
AIME_CAPS   = [2000, 3000, 4000, 5000, 6000, 7000]

_NETWORK_VOLUME = Path("/runpod-volume")
if _NETWORK_VOLUME.exists():
    RESULTS_DIR    = _NETWORK_VOLUME / "early_detection" / "results"
    CHECKPOINT_DIR = _NETWORK_VOLUME / "early_detection" / "checkpoints"
else:
    RESULTS_DIR    = Path("results")
    CHECKPOINT_DIR = Path("checkpoints")

RESULTS_DIR.mkdir(exist_ok=True)


# ── Utilities ─────────────────────────────────────────────────────────────────

def problem_hash(text: str) -> str:
    """SHA-256 of normalised problem text — used for alignment verification."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]


def extract_answer(text: str):
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else None


def is_correct_aime(pred, gold):
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (ValueError, TypeError):
        return pred.strip() == gold.strip()


def is_correct_gsm(pred, gold):
    if pred is None:
        return False
    try:
        return int(float(pred)) == int(float(gold))
    except (ValueError, TypeError):
        return pred.strip() == gold.strip()


def is_correct_math(pred, gold):
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (ValueError, TypeError):
        return pred.strip().lower() == gold.strip().lower()


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + "<think>\n"
    )


def generate_one(tokenizer, model, prompt, max_new_tokens,
                 do_sample=False, temperature=1.0, seed=None,
                 logits_processor=None):
    if seed is not None:
        torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        kwargs["temperature"] = temperature
    if logits_processor:
        kwargs["logits_processor"] = logits_processor
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    generated = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=False)
    return text, len(generated)


def reprompt_for_answer(tokenizer, model, prompt, think_text):
    """Ask model to complete \\boxed{ after forced </think>."""
    think_part  = think_text[:think_text.index("</think>") + len("</think>")]
    ans_prompt  = prompt + think_part + "\n\nThe answer is $\\boxed{"
    ans_text, _ = generate_one(tokenizer, model, ans_prompt,
                               max_new_tokens=ANSWER_CAP, do_sample=False)
    m = re.match(r"([^}]+)", ans_text.strip())
    return m.group(1).strip() if m else None


def independence_baseline(rate_a: float, rate_b: float) -> float:
    return rate_a * (1 - rate_b) + (1 - rate_a) * rate_b


def cohen_kappa(flip_rate: float, baseline: float) -> float:
    if baseline >= 1.0:
        return 0.0
    return 1.0 - flip_rate / baseline


def load_model():
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    if think_end_id == tokenizer.unk_token_id:
        think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
    print(f"  </think> token id : {think_end_id}")
    print(f"  VRAM allocated    : {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return tokenizer, model, think_end_id


# ── Problem loaders ───────────────────────────────────────────────────────────

def load_aime_seed42():
    """200 AIME problems — same shuffle as generate.py / records.json."""
    ds = load_dataset("gneubig/aime-1983-2024", split="train")
    ds = ds.shuffle(seed=42).select(range(200))
    problems = []
    for r in ds:
        problems.append({
            "question": r["Question"],
            "answer":   str(r["Answer"]),
            "id":       r.get("ID", ""),
            "hash":     problem_hash(r["Question"]),
        })
    print(f"  Loaded {len(problems)} AIME problems (seed=42 shuffle)")
    return problems


def load_aime_seed0():
    """200 AIME problems — same as original budget_forcing_fixed_cap.py."""
    ds  = load_dataset("gneubig/aime-1983-2024", split="train")
    rng = random.Random(0)
    sample = rng.sample(list(ds), 200)
    problems = []
    for r in sample:
        problems.append({
            "question": r["Question"],
            "answer":   str(r["Answer"]),
            "id":       r.get("ID", ""),
            "hash":     problem_hash(r["Question"]),
        })
    print(f"  Loaded {len(problems)} AIME problems (seed=0 random.Random)")
    return problems


def load_gsm8k_seed0(n=500):
    """500 GSM8K problems — same as budget_forcing_gsm_math.py."""
    ds  = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(0)
    sample = rng.sample(list(ds), min(n, len(ds)))
    out = []
    for item in sample:
        m = re.search(r"####\s*([\d,\-\.]+)", item["answer"])
        gold = m.group(1).replace(",", "") if m else item["answer"]
        out.append({
            "question": item["question"],
            "answer":   gold,
            "hash":     problem_hash(item["question"]),
        })
    print(f"  Loaded {len(out)} GSM8K problems (seed=0 random.Random)")
    return out


def load_math500_seed0():
    """500 MATH-500 problems — same as budget_forcing_gsm_math.py."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = [{"question": p["problem"], "answer": p["answer"],
            "hash": problem_hash(p["problem"])} for p in ds]
    print(f"  Loaded {len(out)} MATH-500 problems")
    return out


# ── Budget forcing processor ──────────────────────────────────────────────────

class BudgetForcingProcessor(LogitsProcessor):
    def __init__(self, think_end_id: int, budget: int):
        self.think_end_id = think_end_id
        self.budget       = budget
        self.step         = 0
        self.fired        = False

    def __call__(self, input_ids, scores):
        self.step += 1
        if self.step == self.budget and not self.fired:
            self.fired = True
            forced = torch.full_like(scores, float("-inf"))
            forced[:, self.think_end_id] = 0.0
            return forced
        return scores


# ── Part A: GSM8K + MATH-500 forced sweep + uncapped ─────────────────────────

def run_part_a(tokenizer, model, think_end_id):
    out_file = RESULTS_DIR / "final_gsm_math.json"
    print("\n" + "="*60)
    print("PART A: GSM8K + MATH-500 forced sweep + uncapped")
    print(f"  Budgets: {GSM_BUDGETS} + uncapped  |  ANSWER_CAP={ANSWER_CAP}")
    print("="*60)

    gsm_problems  = load_gsm8k_seed0(500)
    math_problems = load_math500_seed0()
    results = {"gsm8k": [], "math500": []}

    for label, problems, correct_fn in [
        ("gsm8k",   gsm_problems,  is_correct_gsm),
        ("math500", math_problems, is_correct_math),
    ]:
        # Forced sweep
        for budget in GSM_BUDGETS:
            print(f"\n  {label} budget={budget}")
            n_correct = n_forced = n_hit_cap = 0
            total_tok = 0
            rows = []

            for i, item in enumerate(problems):
                prompt    = build_prompt(tokenizer, item["question"])
                processor = BudgetForcingProcessor(think_end_id, budget)
                text, n_tokens = generate_one(
                    tokenizer, model, prompt,
                    max_new_tokens=budget + ANSWER_CAP,
                    do_sample=False,
                    logits_processor=[processor],
                )
                converged = "</think>" in text
                answer    = extract_answer(text)

                # Re-prompt if forced and no boxed answer
                if processor.fired and answer is None and converged:
                    answer = reprompt_for_answer(tokenizer, model, prompt, text)

                # Track answer-phase truncation
                answer_phase_tokens = n_tokens - budget if processor.fired else n_tokens
                hit_cap = answer_phase_tokens >= ANSWER_CAP
                if hit_cap: n_hit_cap += 1

                correct = correct_fn(answer, str(item["answer"]))
                if correct:        n_correct += 1
                if processor.fired: n_forced  += 1
                total_tok += n_tokens

                rows.append({"idx": i, "hash": item["hash"],
                             "converged": converged, "correct": correct,
                             "n_tokens": n_tokens, "forced": processor.fired,
                             "hit_answer_cap": hit_cap})

                print(f"  [{i+1:3d}/{len(problems)}] budget={budget} conv={converged} "
                      f"correct={correct} tok={n_tokens} acc={n_correct/(i+1):.3f}")

            n = len(problems)
            entry = {
                "budget":           budget,
                "n_problems":       n,
                "accuracy":         n_correct / n,
                "n_correct":        n_correct,
                "n_forced":         n_forced,
                "n_hit_answer_cap": n_hit_cap,
                "frac_hit_cap":     n_hit_cap / n,
                "mean_tokens":      total_tok / n,
                "rows":             rows,
            }
            results[label].append(entry)
            print(f"  {label} budget={budget}: acc={n_correct/n:.3f}  "
                  f"forced={n_forced}/{n}  hit_cap={n_hit_cap}/{n}")
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)

        # Uncapped baseline — same problem set
        print(f"\n  {label} UNCAPPED")
        n_correct = n_converged = n_hit_cap = 0
        total_tok = 0
        rows = []

        for i, item in enumerate(problems):
            prompt = build_prompt(tokenizer, item["question"])
            text, n_tokens = generate_one(
                tokenizer, model, prompt,
                max_new_tokens=MAX_TOKENS + ANSWER_CAP,
                do_sample=False,
            )
            converged = "</think>" in text
            answer    = extract_answer(text)

            if converged and answer is None:
                answer = reprompt_for_answer(tokenizer, model, prompt, text)

            answer_phase_tokens = n_tokens - MAX_TOKENS if not converged else 0
            hit_cap = answer_phase_tokens >= ANSWER_CAP
            if hit_cap:    n_hit_cap   += 1
            if converged:  n_converged += 1

            correct = correct_fn(answer, str(item["answer"]))
            if correct: n_correct += 1
            total_tok += n_tokens

            rows.append({"idx": i, "hash": item["hash"],
                         "converged": converged, "correct": correct,
                         "n_tokens": n_tokens, "hit_answer_cap": hit_cap})

            print(f"  [{i+1:3d}/{len(problems)}] uncapped conv={converged} "
                  f"correct={correct} tok={n_tokens} acc={n_correct/(i+1):.3f} "
                  f"conv_rate={n_converged/(i+1):.3f}")

        n = len(problems)
        results[label].append({
            "budget":           "uncapped",
            "n_problems":       n,
            "accuracy":         n_correct / n,
            "n_correct":        n_correct,
            "n_converged":      n_converged,
            "convergence_rate": n_converged / n,
            "n_hit_answer_cap": n_hit_cap,
            "frac_hit_cap":     n_hit_cap / n,
            "mean_tokens":      total_tok / n,
            "rows":             rows,
        })
        print(f"  {label} uncapped: acc={n_correct/n:.3f}  conv={n_converged/n:.3f}  "
              f"hit_cap={n_hit_cap}/{n}")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved to {out_file}")
    return results


# ── Part B: AIME fixed caps rerun ─────────────────────────────────────────────

def run_part_b(tokenizer, model, think_end_id):
    out_file = RESULTS_DIR / "final_aime_fixed_cap.json"
    print("\n" + "="*60)
    print("PART B: AIME fixed caps rerun (ANSWER_CAP raised to 1,000)")
    print(f"  Caps: {AIME_CAPS}")
    print("="*60)

    problems = load_aime_seed0()

    # Load uncapped totals from original run (records.json on seed=0 set)
    # Note: records.json uses seed=42. Fixed-cap used seed=0.
    # We rerun uncapped here on seed=0 to get proper reference.
    print("\n  Running uncapped baseline (seed=0) for reference ...")
    n_correct_unc = n_conv_unc = 0
    total_tok_unc = 0
    uncapped_rows = []

    for i, item in enumerate(problems):
        prompt = build_prompt(tokenizer, item["question"])
        text, n_tokens = generate_one(
            tokenizer, model, prompt,
            max_new_tokens=MAX_TOKENS + ANSWER_CAP, do_sample=False,
        )
        converged = "</think>" in text
        answer    = extract_answer(text)
        if converged and answer is None:
            answer = reprompt_for_answer(tokenizer, model, prompt, text)
        correct = is_correct_aime(answer, item["answer"])
        if correct:   n_correct_unc += 1
        if converged: n_conv_unc    += 1
        total_tok_unc += n_tokens
        uncapped_rows.append({"idx": i, "hash": item["hash"],
                              "converged": converged, "correct": correct,
                              "n_tokens": n_tokens})
        print(f"  [{i+1:3d}/200] conv={converged} correct={correct} "
              f"tok={n_tokens} acc={n_correct_unc/(i+1):.3f} "
              f"conv_rate={n_conv_unc/(i+1):.3f}")

    n = len(problems)
    total_tokens_uncapped = total_tok_unc
    uncapped_acc = n_correct_unc / n
    print(f"  Uncapped: acc={uncapped_acc:.3f}  conv={n_conv_unc/n:.3f}  "
          f"total_tok={total_tokens_uncapped}")

    results = {
        "uncapped": {
            "n_problems": n, "accuracy": uncapped_acc,
            "n_correct": n_correct_unc, "n_converged": n_conv_unc,
            "convergence_rate": n_conv_unc / n,
            "total_tokens": total_tokens_uncapped,
            "rows": uncapped_rows,
        },
        "by_cap": [],
    }

    for cap in AIME_CAPS:
        print(f"\n  Cap C={cap}")
        n_correct = n_forced = n_nat_conv = n_hit_cap = 0
        total_tok = 0
        rows = []

        for i, item in enumerate(problems):
            prompt    = build_prompt(tokenizer, item["question"])
            processor = BudgetForcingProcessor(think_end_id, cap)
            text, n_tokens = generate_one(
                tokenizer, model, prompt,
                max_new_tokens=cap + ANSWER_CAP,
                do_sample=False,
                logits_processor=[processor],
            )
            converged = "</think>" in text
            answer    = extract_answer(text)

            if processor.fired and answer is None and converged:
                answer = reprompt_for_answer(tokenizer, model, prompt, text)

            answer_phase_tokens = n_tokens - cap if processor.fired else (
                n_tokens - text[:text.index("</think>") + 8].count("") if converged else 0
            )
            hit_cap = answer_phase_tokens >= ANSWER_CAP

            correct = is_correct_aime(answer, item["answer"])
            if correct:           n_correct  += 1
            if processor.fired:   n_forced   += 1
            if not processor.fired and converged: n_nat_conv += 1
            if hit_cap:           n_hit_cap  += 1
            total_tok += n_tokens

            rows.append({"idx": i, "hash": item["hash"],
                         "converged": converged, "correct": correct,
                         "n_tokens": n_tokens, "forced": processor.fired,
                         "hit_answer_cap": hit_cap})

            print(f"  [{i+1:3d}/200] conv={converged} correct={correct} "
                  f"tok={n_tokens} forced={processor.fired}")

        compute_saved = (total_tokens_uncapped - total_tok) / total_tokens_uncapped
        acc_forced    = sum(
            r["correct"] for r in rows if r["forced"]
        ) / max(n_forced, 1)

        entry = {
            "cap":                cap,
            "n_problems":         n,
            "accuracy":           n_correct / n,
            "n_correct":          n_correct,
            "n_forced":           n_forced,
            "n_naturally_converged": n_nat_conv,
            "accuracy_forced":    acc_forced,
            "compute_saved_rate": compute_saved,
            "n_hit_answer_cap":   n_hit_cap,
            "frac_hit_cap":       n_hit_cap / n,
            "mean_tokens":        total_tok / n,
            "rows":               rows,
        }
        results["by_cap"].append(entry)
        print(f"  C={cap}: acc={n_correct/n:.3f}  saved={compute_saved:.3f}  "
              f"forced={n_forced}/200  hit_cap={n_hit_cap}/200")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved to {out_file}")
    return results


# ── Part C: Temperature rerun on seed=42 problems ────────────────────────────

def run_part_c(tokenizer, model):
    out_file = RESULTS_DIR / "final_temp_rerun.json"
    print("\n" + "="*60)
    print("PART C: Temperature rerun (seed=42 AIME problems)")
    print("="*60)

    problems = load_aime_seed42()

    # Verify alignment with records.json
    with open(CHECKPOINT_DIR / "records.json") as f:
        run1 = json.load(f)
    run1_labels = {r["idx"]: r["converged"] for r in run1}
    print(f"  Run A (records.json): conv={sum(run1_labels.values())}/200")
    # Note: records.json has no hash field; alignment is guaranteed by same shuffle code

    results = {"seeds": {}}

    for seed in TEMP_SEEDS:
        print(f"\n  --- seed={seed} ---")
        rows = []
        n_converged = 0

        for i, item in enumerate(problems):
            prompt = build_prompt(tokenizer, item["question"])
            text, n_tokens = generate_one(
                tokenizer, model, prompt,
                max_new_tokens=MAX_TOKENS + ANSWER_CAP,
                do_sample=True, temperature=TEMPERATURE, seed=seed,
            )
            converged = "</think>" in text
            answer    = extract_answer(text)
            if converged and answer is None:
                answer = reprompt_for_answer(tokenizer, model, prompt, text)
            correct = is_correct_aime(answer, item["answer"])
            if converged: n_converged += 1

            rows.append({
                "idx": i, "problem_id": item["id"], "hash": item["hash"],
                "converged": converged, "correct": correct, "n_tokens": n_tokens,
            })
            print(f"  [{i+1:3d}/200] conv={converged} correct={correct} "
                  f"tok={n_tokens} conv_rate={n_converged/(i+1):.3f}")

        results["seeds"][f"seed_{seed}"] = {
            "convergence_rate": n_converged / 200,
            "n_converged": n_converged,
            "results": rows,
        }
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  seed={seed}: conv={n_converged}/200 = {n_converged/200:.3f}")

    # Majority vote + item-level diff
    print("\n  Computing majority vote and item-level diff ...")
    votes = {}
    for sd in results["seeds"].values():
        for r in sd["results"]:
            votes.setdefault(r["idx"], []).append(r["converged"])

    majority = {idx: (sum(v)/len(v)) >= 0.5 for idx, v in votes.items()}
    n_maj_conv = sum(majority.values())

    flips = sum(run1_labels[i] != majority[i] for i in range(200))
    flip_rate = flips / 200
    rate_a = sum(run1_labels.values()) / 200
    rate_b = n_maj_conv / 200
    indep  = independence_baseline(rate_a, rate_b)
    kappa  = cohen_kappa(flip_rate, indep)

    results["majority_vote"] = {
        "n_converged": n_maj_conv,
        "convergence_rate": rate_b,
    }
    results["item_level_diff"] = {
        "n_problems":            200,
        "n_flips":               flips,
        "flip_rate":             flip_rate,
        "independence_baseline": indep,
        "cohen_kappa":           kappa,
        "greedy_conv_rate":      rate_a,
        "temp_majority_conv_rate": rate_b,
    }

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Majority vote conv rate : {rate_b:.3f}")
    print(f"  Greedy-vs-temp flip rate: {flip_rate:.3f}")
    print(f"  Independence baseline   : {indep:.3f}")
    print(f"  Cohen's kappa           : {kappa:.3f}")
    print(f"\nSaved to {out_file}")
    return results


# ── Part D: Two greedy runs on seed=42 problems ───────────────────────────────

def run_part_d(tokenizer, model):
    out_file = RESULTS_DIR / "final_greedy_rerun.json"
    print("\n" + "="*60)
    print("PART D: Greedy rerun (seed=42 AIME problems)")
    print("="*60)

    problems = load_aime_seed42()

    with open(CHECKPOINT_DIR / "records.json") as f:
        run1 = json.load(f)
    run1_labels = {r["idx"]: r["converged"] for r in run1}
    print(f"  Run A (records.json): conv={sum(run1_labels.values())}/200")

    new_runs = {}

    for run_idx in ["B", "C"]:
        print(f"\n  --- Greedy Run {run_idx} ---")
        rows = []
        n_converged = 0

        for i, item in enumerate(problems):
            prompt = build_prompt(tokenizer, item["question"])
            text, n_tokens = generate_one(
                tokenizer, model, prompt,
                max_new_tokens=MAX_TOKENS + ANSWER_CAP,
                do_sample=False,
            )
            converged = "</think>" in text
            answer    = extract_answer(text)
            if converged and answer is None:
                answer = reprompt_for_answer(tokenizer, model, prompt, text)
            correct = is_correct_aime(answer, item["answer"])
            if converged: n_converged += 1

            rows.append({
                "idx": i, "problem_id": item["id"], "hash": item["hash"],
                "converged": converged, "correct": correct, "n_tokens": n_tokens,
            })
            print(f"  [{i+1:3d}/200] conv={converged} correct={correct} "
                  f"tok={n_tokens} conv_rate={n_converged/(i+1):.3f}")

        new_runs[run_idx] = {r["idx"]: r["converged"] for r in rows}
        print(f"  Run {run_idx}: conv={n_converged}/200 = {n_converged/200:.3f}")

        with open(out_file, "w") as f:
            json.dump({"raw_runs": {k: list(v.items()) for k, v in new_runs.items()}},
                      f, indent=2)

    # Pairwise agreement
    all_runs = {"A": run1_labels, **new_runs}
    pairs = [("A","B","runA_vs_runB"), ("A","C","runA_vs_runC"), ("B","C","runB_vs_runC")]
    pairwise = []
    for a, b, name in pairs:
        common = sorted(set(all_runs[a]) & set(all_runs[b]))
        agree  = sum(all_runs[a][i] == all_runs[b][i] for i in common)
        flips  = len(common) - agree
        rate_a = sum(all_runs[a].values()) / len(all_runs[a])
        rate_b = sum(all_runs[b].values()) / len(all_runs[b])
        indep  = independence_baseline(rate_a, rate_b)
        kappa  = cohen_kappa(flips / len(common), indep)
        pairwise.append({
            "comparison":            name,
            "n_common":              len(common),
            "agreement":             agree / len(common),
            "flip_rate":             flips / len(common),
            "n_flips":               flips,
            "independence_baseline": indep,
            "cohen_kappa":           kappa,
        })

    mean_flip = sum(p["flip_rate"] for p in pairwise) / len(pairwise)

    output = {
        "pairwise": pairwise,
        "mean_greedy_flip_rate": mean_flip,
        "raw_runs": {k: list(v.items()) for k, v in new_runs.items()},
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*50}\nPAIRWISE GREEDY AGREEMENT\n{'='*50}")
    for p in pairwise:
        print(f"  {p['comparison']}: agreement={p['agreement']:.3f}  "
              f"flip={p['flip_rate']:.3f}  baseline={p['independence_baseline']:.3f}  "
              f"kappa={p['cohen_kappa']:.3f}  ({p['n_flips']}/200 flips)")
    print(f"  Mean flip rate: {mean_flip:.3f}")
    print(f"\nSaved to {out_file}")
    return output


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", default=["A", "B", "C", "D"],
                        choices=["A", "B", "C", "D"])
    args = parser.parse_args()

    tokenizer, model, think_end_id = load_model()

    if "A" in args.parts:
        run_part_a(tokenizer, model, think_end_id)
    if "B" in args.parts:
        run_part_b(tokenizer, model, think_end_id)
    if "C" in args.parts:
        run_part_c(tokenizer, model)
    if "D" in args.parts:
        run_part_d(tokenizer, model)

    print("\n\nAll parts complete.")


if __name__ == "__main__":
    main()
