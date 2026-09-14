"""
Budget-forcing sweep at 32/64/128/256 tokens on GSM8K and MATH-500 (7B, greedy).

Establishes the actual B* for the two easy benchmarks claimed in the paper.
The existing budget_sweep_extended.py only tested natural convergence starting
at budget=256; this script provides the sub-256 forced evidence.

Outputs: results/budget_forcing_gsm_math.json

Usage: python budget_forcing_gsm_math.py
"""

import json
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

MODEL_NAME  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
BUDGETS     = [32, 64, 128, 256]
N_GSM8K     = 500   # random sample — full set is 1319 test problems
N_MATH500   = 500   # full MATH-500 test set
SEED        = 0

_NETWORK_VOLUME = Path("/runpod-volume")
if _NETWORK_VOLUME.exists():
    RESULTS_DIR = _NETWORK_VOLUME / "early_detection" / "results"
else:
    RESULTS_DIR = Path("results")

OUT_FILE = RESULTS_DIR / "budget_forcing_gsm_math.json"


class BudgetForcingProcessor(LogitsProcessor):
    def __init__(self, think_end_id: int, budget: int):
        self.think_end_id = think_end_id
        self.budget       = budget
        self.step         = 0
        self.fired        = False

    def __call__(self, input_ids, scores):
        self.step += 1
        if self.step == self.budget:
            self.fired = True
            forced = torch.full_like(scores, float("-inf"))
            forced[:, self.think_end_id] = 0.0
            return forced
        return scores


def extract_answer(text: str):
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else None


def is_correct_gsm(pred: str, gold: str) -> bool:
    if pred is None:
        return False
    # GSM8K answers are integers
    try:
        return int(float(pred)) == int(float(gold))
    except (ValueError, TypeError):
        return pred.strip() == gold.strip()


def is_correct_math(pred: str, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (ValueError, TypeError):
        return pred.strip().lower() == gold.strip().lower()


def load_model():
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto"
    )
    model.eval()
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    if think_end_id == tokenizer.unk_token_id:
        think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
    print(f"  </think> token id : {think_end_id}")
    print(f"  VRAM allocated    : {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return tokenizer, model, think_end_id


def load_gsm8k(n: int) -> list:
    ds  = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(SEED)
    problems = rng.sample(list(ds), min(n, len(ds)))
    # Normalize: extract final numeric answer from solution string
    out = []
    for item in problems:
        ans_match = re.search(r"####\s*([\d,\-\.]+)", item["answer"])
        gold = ans_match.group(1).replace(",", "") if ans_match else item["answer"]
        out.append({"question": item["question"], "answer": gold})
    return out


def load_math500() -> list:
    # hendrycks/competition_math filtered to MATH-500 subset used in literature
    # Use lighteval/MATH split which matches the 500-problem test set
    try:
        ds = load_dataset("lighteval/MATH", "all", split="test")
        rng = random.Random(SEED)
        problems = rng.sample(list(ds), min(N_MATH500, len(ds)))
        return [{"question": p["problem"], "answer": p["solution"]} for p in problems]
    except Exception:
        # Fallback: hendrycks MATH
        ds = load_dataset("hendrycks/competition_math", split="test")
        rng = random.Random(SEED)
        problems = rng.sample(list(ds), min(N_MATH500, len(ds)))
        return [{"question": p["problem"], "answer": p["solution"]} for p in problems]


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + "<think>\n"
    )


def run_budget(budget, problems, is_correct_fn, tokenizer, model, think_end_id, device, label):
    print(f"\n  Budget={budget} | {label} ({len(problems)} problems)")
    n_correct = 0
    n_forced  = 0
    total_tok = 0

    for i, item in enumerate(problems):
        prompt    = build_prompt(tokenizer, item["question"])
        inputs    = tokenizer(prompt, return_tensors="pt").to(device)
        processor = BudgetForcingProcessor(think_end_id, budget)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=budget + 256,
                do_sample=False,
                logits_processor=[processor],
            )

        generated = out[0][inputs["input_ids"].shape[1]:]
        text      = tokenizer.decode(generated, skip_special_tokens=False)
        n_tokens  = len(generated)
        total_tok += n_tokens

        answer = extract_answer(text)

        # Re-prompt if forced and no boxed answer
        if processor.fired and answer is None and "</think>" in text:
            think_part   = text[:text.index("</think>") + len("</think>")]
            answer_prompt = prompt + think_part + "\n\nThe answer is $\\boxed{"
            ans_inputs   = tokenizer(answer_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                ans_out = model.generate(
                    **ans_inputs, max_new_tokens=32, do_sample=False,
                )
            ans_text = tokenizer.decode(
                ans_out[0][ans_inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            m = re.match(r"([^}]+)", ans_text.strip())
            answer = m.group(1).strip() if m else None

        correct = is_correct_fn(answer, str(item["answer"]))
        if processor.fired:
            n_forced += 1
        if correct:
            n_correct += 1

        if (i + 1) % 50 == 0:
            print(f"    [{i+1:3d}/{len(problems)}] running acc={n_correct/(i+1):.3f}")

    n = len(problems)
    return {
        "budget":           budget,
        "n_problems":       n,
        "accuracy":         n_correct / n,
        "n_correct":        n_correct,
        "n_forced":         n_forced,
        "mean_tokens":      total_tok / n,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model, think_end_id = load_model()

    print("\nLoading datasets ...")
    gsm_problems  = load_gsm8k(N_GSM8K)
    math_problems = load_math500()
    print(f"  GSM8K : {len(gsm_problems)} problems")
    print(f"  MATH  : {len(math_problems)} problems")

    results = {"gsm8k": [], "math500": []}

    print("\n=== GSM8K ===")
    for budget in BUDGETS:
        r = run_budget(budget, gsm_problems, is_correct_gsm,
                       tokenizer, model, think_end_id, device, "GSM8K")
        results["gsm8k"].append(r)
        with open(OUT_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  GSM8K  budget={budget}: acc={r['accuracy']:.3f}  forced={r['n_forced']}/{r['n_problems']}")

    print("\n=== MATH-500 ===")
    for budget in BUDGETS:
        r = run_budget(budget, math_problems, is_correct_math,
                       tokenizer, model, think_end_id, device, "MATH-500")
        results["math500"].append(r)
        with open(OUT_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  MATH500 budget={budget}: acc={r['accuracy']:.3f}  forced={r['n_forced']}/{r['n_problems']}")

    # Summary table
    print(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}")
    for bench, key in [("GSM8K", "gsm8k"), ("MATH-500", "math500")]:
        print(f"\n{bench}")
        print(f"  {'Budget':>7}  {'Acc':>6}  {'Forced':>7}  {'MeanTok':>8}")
        for r in results[key]:
            print(f"  {r['budget']:6d}  {r['accuracy']:6.3f}  "
                  f"{r['n_forced']:6d}/{r['n_problems']}  {r['mean_tokens']:8.0f}")

    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
