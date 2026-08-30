"""
7B Budget-Forcing at small budgets: 32, 64, 128 tokens.
Same 200 AIME problems as original run. Greedy decoding (k=1).
Forces </think> at each budget, then extracts answer.

Outputs: results/budget_forcing_7B_extended.json

Usage: python budget_forcing_7B_extended.py
"""

import json
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
N_PROBLEMS = 200
BUDGETS    = [32, 64, 128]
MAX_TOKENS = 10_000

_NETWORK_VOLUME = Path("/runpod-volume")
if _NETWORK_VOLUME.exists():
    RESULTS_DIR = _NETWORK_VOLUME / "early_detection" / "results"
else:
    RESULTS_DIR = Path("results")

OUT_FILE = RESULTS_DIR / "budget_forcing_7B_extended.json"


class BudgetForcingProcessor(LogitsProcessor):
    def __init__(self, think_end_id, budget):
        self.think_end_id = think_end_id
        self.budget = budget
        self.step = 0

    def __call__(self, input_ids, scores):
        self.step += 1
        if self.step == self.budget:
            forced = torch.full_like(scores, float("-inf"))
            forced[:, self.think_end_id] = 0.0
            return forced
        return scores


def extract_answer(text):
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else None


def is_correct(pred, gold):
    if pred is None:
        return False
    try:
        return float(pred) == float(gold)
    except ValueError:
        return pred.strip().lower() == str(gold).strip().lower()


def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto"
    )
    model.eval()
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    if think_end_id == tokenizer.unk_token_id:
        think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
    print(f"  </think> token id: {think_end_id}")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return tokenizer, model, think_end_id


def load_problems():
    ds = load_dataset("gneubig/aime-1983-2024", split="train")
    rng = random.Random(0)
    return rng.sample(list(ds), N_PROBLEMS)


def run_budget(budget, problems, tokenizer, model, think_end_id, device, prompt):
    print(f"\n{'='*60}\nBudget = {budget} tokens\n{'='*60}")
    results = []

    for i, item in enumerate(problems):
        gold    = str(item["Answer"])
        prob_id = item.get("ID", i)

        messages = [{"role": "user", "content": item["Question"]}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + "<think>\n"

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        processor = BudgetForcingProcessor(think_end_id, budget)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=budget + 200,
                do_sample=False,
                logits_processor=[processor],
            )

        generated = out[0][inputs["input_ids"].shape[1]:]
        text      = tokenizer.decode(generated, skip_special_tokens=False)
        n_tokens  = len(generated)
        converged = "</think>" in text
        answer    = extract_answer(text)

        # Re-prompt for answer if not found after forced </think>
        if converged and answer is None:
            think_part   = text[:text.index("</think>") + len("</think>")]
            answer_prompt = prompt_text + think_part + "\n\nThe answer is $\\boxed{"
            ans_inputs   = tokenizer(answer_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                ans_out = model.generate(
                    **ans_inputs, max_new_tokens=20, do_sample=False,
                )
            ans_text = tokenizer.decode(
                ans_out[0][ans_inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            m = re.match(r"([^}]+)", ans_text.strip())
            answer = m.group(1).strip() if m else None

        correct = is_correct(answer, gold)
        results.append({
            "problem_id": prob_id,
            "budget":     budget,
            "converged":  converged,
            "n_tokens":   n_tokens,
            "correct":    correct,
            "pred":       answer,
            "gold":       gold,
        })
        print(f"  [{i+1:3d}/{N_PROBLEMS}] converged={converged} tokens={n_tokens:4d} correct={correct} pred={answer} gold={gold}")

    n_conv    = sum(r["converged"] for r in results)
    n_correct = sum(r["correct"]   for r in results)
    mean_tok  = sum(r["n_tokens"]  for r in results) / len(results)
    n_conv_correct = sum(r["correct"] for r in results if r["converged"])

    print(f"\n  Budget {budget}: conv={n_conv/N_PROBLEMS:.3f} acc={n_correct/N_PROBLEMS:.3f} "
          f"conv_acc={n_conv_correct/max(n_conv,1):.3f} mean_tok={mean_tok:.0f}")

    return results, {
        "budget":               budget,
        "n_problems":           N_PROBLEMS,
        "convergence_rate":     n_conv / N_PROBLEMS,
        "n_converged":          n_conv,
        "accuracy":             n_correct / N_PROBLEMS,
        "n_correct":            n_correct,
        "converged_accuracy":   n_conv_correct / max(n_conv, 1),
        "mean_tokens_generated": mean_tok,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, think_end_id = load_model()
    problems = load_problems()
    print(f"Loaded {len(problems)} problems")

    all_items  = []
    by_budget  = []

    for budget in BUDGETS:
        items, summary = run_budget(budget, problems, tokenizer, model, think_end_id, device, "")
        all_items.extend(items)
        by_budget.append(summary)

        with open(OUT_FILE, "w") as f:
            json.dump({"model": MODEL_NAME, "budgets": BUDGETS,
                       "by_budget": by_budget, "results": all_items}, f, indent=2)

    print(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}")
    print(f"{'Budget':>8}  {'Conv':>6}  {'Acc':>6}  {'ConvAcc':>8}  {'MeanTok':>8}")
    for s in by_budget:
        print(f"  {s['budget']:6d}  {s['convergence_rate']:6.3f}  {s['accuracy']:6.3f}  "
              f"{s['converged_accuracy']:8.3f}  {s['mean_tokens_generated']:8.0f}")
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
