"""
Combined rerun script — fixes three alignment issues flagged in review.

Part A: Uncapped GSM8K and MATH-500 baselines
  - Same 500 problems used in budget_forcing_gsm_math.py (random.Random(0).sample)
  - Provides valid uncapped baselines to compare against forced-sweep numbers

Part B: Temperature runs on correct AIME problem set
  - Uses ds.shuffle(seed=42).select(range(200)) — same as generate.py / records.json
  - 4 seeds (0-3), T=0.6, majority vote per problem
  - Provides valid item-level greedy-vs-temperature comparison

Part C: Two additional greedy runs on correct AIME problem set
  - Same seed=42 shuffle as records.json
  - Provides valid pairwise greedy item-level agreement

Outputs:
  results/uncapped_gsm_math.json
  results/temp_rerun.json
  results/greedy_rerun.json

Usage: python combined_rerun.py [--parts A B C]
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_TOKENS  = 10_000
ANSWER_CAP  = 1_000   # raised from 200 to avoid answer truncation (item 3)
SEEDS       = [0, 1, 2, 3]
TEMPERATURE = 0.6

_NETWORK_VOLUME = Path("/runpod-volume")
if _NETWORK_VOLUME.exists():
    RESULTS_DIR    = _NETWORK_VOLUME / "early_detection" / "results"
    CHECKPOINT_DIR = _NETWORK_VOLUME / "early_detection" / "checkpoints"
else:
    RESULTS_DIR    = Path("results")
    CHECKPOINT_DIR = Path("checkpoints")

RESULTS_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    print(f"  </think> token id: {think_end_id}")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return tokenizer, model, think_end_id


def extract_answer(text):
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


def build_prompt(tokenizer, question):
    messages = [{"role": "user", "content": question}]
    return (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + "<think>\n"
    )


def generate_one(tokenizer, model, prompt, max_new_tokens, do_sample=False,
                 temperature=1.0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    if do_sample:
        kwargs["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    generated = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=False)
    return text, len(generated)


# ── Problem loaders ───────────────────────────────────────────────────────────

def load_aime_seed42():
    """Same 200 problems as generate.py / records.json."""
    ds = load_dataset("gneubig/aime-1983-2024", split="train")
    ds = ds.shuffle(seed=42).select(range(200))
    problems = []
    for r in ds:
        problems.append({
            "question": r["Question"],
            "answer":   str(r["Answer"]),
            "id":       r.get("ID", ""),
        })
    return problems


def load_gsm8k_seed0(n=500):
    """Same 500 problems as budget_forcing_gsm_math.py."""
    ds  = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(0)
    sample = rng.sample(list(ds), min(n, len(ds)))
    out = []
    for item in sample:
        m = re.search(r"####\s*([\d,\-\.]+)", item["answer"])
        gold = m.group(1).replace(",", "") if m else item["answer"]
        out.append({"question": item["question"], "answer": gold})
    return out


def load_math500_seed0():
    """Same 500 problems as budget_forcing_gsm_math.py."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [{"question": p["problem"], "answer": p["answer"]} for p in ds]


# ── Part A: Uncapped GSM8K + MATH-500 ────────────────────────────────────────

def run_part_a(tokenizer, model):
    out_file = RESULTS_DIR / "uncapped_gsm_math.json"
    print("\n" + "="*60)
    print("PART A: Uncapped GSM8K + MATH-500 baselines")
    print("="*60)

    gsm_problems  = load_gsm8k_seed0(500)
    math_problems = load_math500_seed0()
    print(f"  GSM8K:    {len(gsm_problems)} problems")
    print(f"  MATH-500: {len(math_problems)} problems")

    results = {}

    for label, problems, correct_fn in [
        ("gsm8k",   gsm_problems,  is_correct_gsm),
        ("math500", math_problems, is_correct_math),
    ]:
        print(f"\n  --- {label} uncapped ---")
        n_correct = 0
        n_converged = 0
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
            correct   = correct_fn(answer, str(item["answer"]))

            if converged and answer is None:
                think_part   = text[:text.index("</think>") + len("</think>")]
                ans_prompt   = prompt + think_part + "\n\nThe answer is $\\boxed{"
                ans_text, _  = generate_one(
                    tokenizer, model, ans_prompt,
                    max_new_tokens=ANSWER_CAP, do_sample=False,
                )
                m = re.match(r"([^}]+)", ans_text.strip())
                answer  = m.group(1).strip() if m else None
                correct = correct_fn(answer, str(item["answer"]))

            if correct:   n_correct   += 1
            if converged: n_converged += 1
            total_tok += n_tokens

            rows.append({"idx": i, "converged": converged, "correct": correct,
                         "n_tokens": n_tokens})

            if (i + 1) % 50 == 0:
                print(f"    [{i+1:3d}/{len(problems)}] acc={n_correct/(i+1):.3f} "
                      f"conv={n_converged/(i+1):.3f}")

        n = len(problems)
        results[label] = {
            "n_problems":       n,
            "n_correct":        n_correct,
            "accuracy":         n_correct / n,
            "n_converged":      n_converged,
            "convergence_rate": n_converged / n,
            "mean_tokens":      total_tok / n,
            "rows":             rows,
        }
        print(f"  {label}: acc={n_correct/n:.3f}  conv={n_converged/n:.3f}")

        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved to {out_file}")
    return results


# ── Part B: Temperature rerun on seed=42 AIME problems ───────────────────────

def run_part_b(tokenizer, model):
    out_file = RESULTS_DIR / "temp_rerun.json"
    print("\n" + "="*60)
    print("PART B: Temperature rerun (seed=42 AIME problems)")
    print("="*60)

    problems = load_aime_seed42()
    print(f"  Loaded {len(problems)} AIME problems (seed=42 shuffle)")

    results = {"seeds": {}, "majority_vote": {}}

    for seed in SEEDS:
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
            correct   = is_correct_aime(answer, item["answer"])

            if converged and answer is None:
                think_part  = text[:text.index("</think>") + len("</think>")]
                ans_prompt  = prompt + think_part + "\n\nThe answer is $\\boxed{"
                ans_text, _ = generate_one(
                    tokenizer, model, ans_prompt,
                    max_new_tokens=ANSWER_CAP, do_sample=False,
                )
                m = re.match(r"([^}]+)", ans_text.strip())
                answer  = m.group(1).strip() if m else None
                correct = is_correct_aime(answer, item["answer"])

            if converged: n_converged += 1
            rows.append({
                "idx": i, "problem_id": item["id"],
                "converged": converged, "correct": correct, "n_tokens": n_tokens,
            })
            print(f"  [{i+1:3d}/200] conv={converged} correct={correct} "
                  f"tok={n_tokens} conv_rate={n_converged/(i+1):.3f}")

        n = len(problems)
        results["seeds"][f"seed_{seed}"] = {
            "convergence_rate": n_converged / n,
            "n_converged": n_converged,
            "results": rows,
        }
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  seed={seed} done: conv={n_converged/n:.3f}")

    # Majority vote
    print("\n  Computing majority vote ...")
    votes = {}
    for seed_data in results["seeds"].values():
        for r in seed_data["results"]:
            votes.setdefault(r["idx"], []).append(r["converged"])

    majority = {idx: (sum(v)/len(v)) >= 0.5 for idx, v in votes.items()}
    n_maj_conv = sum(majority.values())

    # Item-level vs records.json greedy (seed=42 set)
    with open(CHECKPOINT_DIR / "records.json") as f:
        greedy = json.load(f)
    greedy_labels = {r["idx"]: r["converged"] for r in greedy}

    flips = sum(greedy_labels[i] != majority[i] for i in range(len(problems)))
    flip_rate = flips / len(problems)
    indep_baseline = (
        sum(greedy_labels.values()) / len(greedy_labels) *
        (1 - n_maj_conv / len(problems)) +
        (1 - sum(greedy_labels.values()) / len(greedy_labels)) *
        n_maj_conv / len(problems)
    )

    results["majority_vote"] = {
        "n_converged":      n_maj_conv,
        "convergence_rate": n_maj_conv / len(problems),
        "majority_labels":  majority,
    }
    results["item_level_diff"] = {
        "n_problems":        len(problems),
        "flip_rate":         flip_rate,
        "n_flips":           flips,
        "independence_baseline": indep_baseline,
        "cohen_kappa":       1 - flip_rate / indep_baseline if indep_baseline > 0 else None,
    }

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Majority vote conv rate: {n_maj_conv/len(problems):.3f}")
    print(f"  Greedy-vs-temp flip rate: {flip_rate:.3f}")
    print(f"  Independence baseline:    {indep_baseline:.3f}")
    print(f"  Cohen's kappa:            {results['item_level_diff']['cohen_kappa']:.3f}")
    print(f"\nSaved to {out_file}")
    return results


# ── Part C: Two greedy runs on seed=42 AIME problems ─────────────────────────

def run_part_c(tokenizer, model):
    out_file = RESULTS_DIR / "greedy_rerun.json"
    print("\n" + "="*60)
    print("PART C: Greedy rerun (seed=42 AIME problems)")
    print("="*60)

    problems = load_aime_seed42()
    print(f"  Loaded {len(problems)} AIME problems (seed=42 shuffle)")

    # Load Run 1 (records.json)
    with open(CHECKPOINT_DIR / "records.json") as f:
        run1_records = json.load(f)
    run1_labels = {r["idx"]: r["converged"] for r in run1_records}
    print(f"  Run 1 (records.json): conv={sum(run1_labels.values())}/200")

    new_runs = {}

    for run_idx in [2, 3]:
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
            correct   = is_correct_aime(answer, item["answer"])

            if converged: n_converged += 1
            rows.append({
                "idx": i, "problem_id": item["id"],
                "converged": converged, "correct": correct, "n_tokens": n_tokens,
            })
            print(f"  [{i+1:3d}/200] conv={converged} correct={correct} "
                  f"tok={n_tokens} conv_rate={n_converged/(i+1):.3f}")

        new_runs[run_idx] = {r["idx"]: r["converged"] for r in rows}
        print(f"  Run {run_idx}: conv={n_converged}/200 = {n_converged/200:.3f}")

        with open(out_file, "w") as f:
            json.dump({"raw_runs": {str(k): v for k, v in new_runs.items()}}, f, indent=2)

    # Pairwise agreement
    all_runs = {1: run1_labels, **new_runs}
    pairs = [(1,2,"run1_vs_run2"), (1,3,"run1_vs_run3"), (2,3,"run2_vs_run3")]
    pairwise = []
    for a, b, name in pairs:
        common = sorted(set(all_runs[a]) & set(all_runs[b]))
        agree  = sum(all_runs[a][i] == all_runs[b][i] for i in common)
        flips  = len(common) - agree
        indep  = (sum(all_runs[a].values())/len(all_runs[a]) *
                  (1 - sum(all_runs[b].values())/len(all_runs[b])) +
                  (1 - sum(all_runs[a].values())/len(all_runs[a])) *
                  sum(all_runs[b].values())/len(all_runs[b]))
        pairwise.append({
            "comparison": name,
            "n_common":   len(common),
            "agreement":  agree / len(common),
            "flip_rate":  flips / len(common),
            "n_flips":    flips,
            "independence_baseline": indep,
        })

    mean_flip = sum(p["flip_rate"] for p in pairwise) / len(pairwise)

    output = {
        "pairwise": pairwise,
        "mean_greedy_flip_rate": mean_flip,
        "raw_runs": {str(k): list(new_runs[k].items()) for k in new_runs},
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*50}\nPAIRWISE GREEDY AGREEMENT\n{'='*50}")
    for p in pairwise:
        print(f"  {p['comparison']}: agreement={p['agreement']:.3f}  "
              f"flip_rate={p['flip_rate']:.3f}  baseline={p['independence_baseline']:.3f}  "
              f"({p['n_flips']}/{p['n_common']} flips)")
    print(f"  Mean pairwise flip rate: {mean_flip:.3f}")
    print(f"\nSaved to {out_file}")
    return output


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C"],
                        help="Which parts to run (default: all)")
    args = parser.parse_args()

    tokenizer, model, _ = load_model()

    if "A" in args.parts:
        run_part_a(tokenizer, model)
    if "B" in args.parts:
        run_part_b(tokenizer, model)
    if "C" in args.parts:
        run_part_c(tokenizer, model)

    print("\n\nAll done.")


if __name__ == "__main__":
    main()
