"""
Budget-Forcing Experiment — DeepSeek-R1-Distill-Qwen-1.5B

Same 200 AIME problems as the 7B run. Greedy decoding (deterministic, 1 seed).
Forces </think> at each token budget; runs to 10k ceiling for non-converged.

Outputs:
  results/budget_forcing_1.5B.json        — full item-level results
  results/1.5B_convergence_rates.json     — convergence + accuracy by budget

Usage:
  python budget_forcing_1_5B.py
"""

import json
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
N_PROBLEMS   = 200
BUDGETS      = [256, 512, 1024, 10_000]
MAX_TOKENS   = 10_000
THINK_END    = "</think>"

_NETWORK_VOLUME = Path("/runpod-volume")
if _NETWORK_VOLUME.exists():
    RESULTS_DIR = _NETWORK_VOLUME / "early_detection" / "results"
else:
    RESULTS_DIR = Path("results")

OUT_MAIN = RESULTS_DIR / "budget_forcing_1.5B.json"
OUT_CONV = RESULTS_DIR / "1.5B_convergence_rates.json"


# ── Budget-forcing logits processor ───────────────────────────────────────────
class BudgetForcingProcessor(LogitsProcessor):
    """
    Forces the </think> token at exactly `budget` generated tokens by
    setting all other logits to -inf at that step.
    """
    def __init__(self, think_end_id: int, budget: int):
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


# ── Answer extraction ──────────────────────────────────────────────────────────
def extract_answer(text: str):
    """Extract the final boxed answer from model output."""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else None


def is_correct(pred, gold):
    if pred is None:
        return False
    try:
        return float(pred) == float(gold)
    except ValueError:
        return pred.strip().lower() == str(gold).strip().lower()


# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # Find </think> token id
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    if think_end_id == tokenizer.unk_token_id:
        # Fallback: encode and take first token
        think_end_id = tokenizer.encode(THINK_END, add_special_tokens=False)[0]
    print(f"  </think> token id: {think_end_id}")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return tokenizer, model, think_end_id


# ── Load problems ─────────────────────────────────────────────────────────────
def load_problems():
    ds = load_dataset("gneubig/aime-1983-2024", split="train")
    all_problems = list(ds)
    rng = random.Random(0)  # same seed-0 draw as temp_robustness.py
    return rng.sample(all_problems, N_PROBLEMS)


# ── Run one budget ────────────────────────────────────────────────────────────
def run_budget(budget, problems, tokenizer, model, think_end_id, device):
    print(f"\n{'='*60}")
    print(f"Budget = {budget} tokens")
    print(f"{'='*60}")

    results = []
    for i, item in enumerate(problems):
        problem  = item["Question"]
        gold     = str(item["Answer"])
        prob_id  = item.get("ID", i)

        messages = [{"role": "user", "content": problem}]
        prompt   = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + "<think>\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        if budget < MAX_TOKENS:
            processor = BudgetForcingProcessor(think_end_id, budget)
            logits_processors = [processor]
            max_new = budget + 200  # extra tokens for the answer after </think>
        else:
            logits_processors = []
            max_new = MAX_TOKENS

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,       # greedy
                logits_processor=logits_processors,
            )

        generated = out[0][inputs["input_ids"].shape[1]:]
        text      = tokenizer.decode(generated, skip_special_tokens=False)
        n_tokens  = len(generated)
        converged = THINK_END in text

        # If budget-forced, the model may not produce \boxed{} naturally.
        # Re-prompt with the partial generation + </think> to elicit an answer.
        if budget < MAX_TOKENS and converged and extract_answer(text) is None:
            think_part = text[:text.index(THINK_END) + len(THINK_END)]
            answer_prompt = prompt + think_part + "\n\nThe answer is $\\boxed{"
            ans_inputs = tokenizer(answer_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                ans_out = model.generate(
                    **ans_inputs, max_new_tokens=20, do_sample=False,
                )
            ans_text = tokenizer.decode(
                ans_out[0][ans_inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            # Extract number before closing brace
            m = re.match(r"([^}]+)", ans_text.strip())
            answer = m.group(1).strip() if m else None
        else:
            answer = extract_answer(text)
        correct   = is_correct(answer, gold)

        results.append({
            "problem_id":  prob_id,
            "budget":      budget,
            "converged":   converged,
            "n_tokens":    n_tokens,
            "correct":     correct,
            "pred_answer": answer,
            "gold_answer": gold,
        })

        print(
            f"  [{i+1:3d}/{N_PROBLEMS}] budget={budget:5d} "
            f"converged={converged} tokens={n_tokens:5d} "
            f"correct={correct} pred={answer} gold={gold}"
        )

    n_conv    = sum(r["converged"] for r in results)
    n_correct = sum(r["correct"]   for r in results)
    n_conv_correct = sum(r["correct"] for r in results if r["converged"])
    mean_tok  = sum(r["n_tokens"]  for r in results) / len(results)

    print(f"\n  Budget {budget}: conv={n_conv/N_PROBLEMS:.3f} "
          f"acc={n_correct/N_PROBLEMS:.3f} "
          f"conv_acc={n_conv_correct/max(n_conv,1):.3f} "
          f"mean_tokens={mean_tok:.0f}")

    return results, {
        "budget":                   budget,
        "n_problems":               N_PROBLEMS,
        "convergence_rate":         n_conv / N_PROBLEMS,
        "n_converged":              n_conv,
        "accuracy":                 n_correct / N_PROBLEMS,
        "n_correct":                n_correct,
        "converged_accuracy":       n_conv_correct / max(n_conv, 1),
        "mean_tokens_generated":    mean_tok,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model, think_end_id = load_model()
    problems = load_problems()
    print(f"Loaded {len(problems)} problems")

    all_item_results = []
    conv_summary     = []

    for budget in BUDGETS:
        item_results, summary = run_budget(
            budget, problems, tokenizer, model, think_end_id, device
        )
        all_item_results.extend(item_results)
        conv_summary.append(summary)

        # Save incrementally
        with open(OUT_MAIN, "w") as f:
            json.dump({
                "model":    MODEL_NAME,
                "budgets":  BUDGETS,
                "results":  all_item_results,
            }, f, indent=2)
        with open(OUT_CONV, "w") as f:
            json.dump({"model": MODEL_NAME, "by_budget": conv_summary}, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Budget':>8}  {'Conv':>6}  {'Acc':>6}  {'ConvAcc':>8}  {'MeanTok':>8}")
    for s in conv_summary:
        print(f"  {s['budget']:6d}  {s['convergence_rate']:6.3f}  "
              f"{s['accuracy']:6.3f}  {s['converged_accuracy']:8.3f}  "
              f"{s['mean_tokens_generated']:8.0f}")

    print(f"\nSaved to {OUT_MAIN}")
    print(f"         {OUT_CONV}")


if __name__ == "__main__":
    main()
