"""
Budget-forcing fixed-cap experiment on AIME (7B, greedy).

For each cap C in CAPS, forces </think> at token C if not already emitted,
then lets the model generate a boxed answer. Reports:
  - compute_saved_rate : fraction of total uncapped tokens NOT spent
  - accuracy           : fraction of 200 problems answered correctly
  - accuracy_converged : accuracy among generations that converged naturally (before C)
  - accuracy_forced    : accuracy among generations forced at C

This replaces the broken "net savings" metric in early_exit_analysis.py.
The correct comparison baseline is uncapped inference (total_tokens_no_cap).

Outputs: results/budget_forcing_fixed_cap.json

Usage: python budget_forcing_fixed_cap.py
"""

import json
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
N_PROBLEMS  = 200
CAPS        = [2000, 3000, 4000, 5000, 6000, 7000]
MAX_TOKENS  = 10_000          # uncapped ceiling (matches original run)
SEED        = 0

_NETWORK_VOLUME = Path("/runpod-volume")
if _NETWORK_VOLUME.exists():
    RESULTS_DIR = _NETWORK_VOLUME / "early_detection" / "results"
    CHECKPOINT_DIR = _NETWORK_VOLUME / "early_detection" / "checkpoints"
else:
    RESULTS_DIR    = Path("results")
    CHECKPOINT_DIR = Path("checkpoints")

OUT_FILE = RESULTS_DIR / "budget_forcing_fixed_cap.json"


class BudgetForcingProcessor(LogitsProcessor):
    """Force </think> exactly at step `budget` if not already emitted."""
    def __init__(self, think_end_id: int, budget: int):
        self.think_end_id = think_end_id
        self.budget       = budget
        self.step         = 0
        self.fired        = False   # True if we actually forced </think>

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


def is_correct(pred, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred.strip().lower() == gold.strip().lower()


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


def load_problems():
    ds  = load_dataset("gneubig/aime-1983-2024", split="train")
    rng = random.Random(SEED)
    return rng.sample(list(ds), N_PROBLEMS)


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + "<think>\n"
    )


def run_one(cap: int, problems, tokenizer, model, think_end_id, device):
    """Run all 200 problems under budget-forcing cap C."""
    print(f"\n{'='*60}\nCap = {cap} tokens\n{'='*60}")
    records = []

    for i, item in enumerate(problems):
        gold    = str(item["Answer"])
        prob_id = item.get("ID", i)
        prompt  = build_prompt(tokenizer, item["Question"])

        inputs    = tokenizer(prompt, return_tensors="pt").to(device)
        processor = BudgetForcingProcessor(think_end_id, cap)
        device    = next(model.parameters()).device

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=cap + 256,   # room for </think> + boxed answer
                do_sample=False,
                logits_processor=[processor],
            )

        generated = out[0][inputs["input_ids"].shape[1]:]
        text      = tokenizer.decode(generated, skip_special_tokens=False)
        n_tokens  = len(generated)

        # processor.fired is True only if we injected </think> at step==cap
        was_forced          = processor.fired
        naturally_converged = (not was_forced) and ("</think>" in text)
        answer              = extract_answer(text)

        # Re-prompt for boxed answer when forced </think> didn't produce one
        if was_forced and answer is None:
            think_part   = text[:text.index("</think>") + len("</think>")] if "</think>" in text else text
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

        correct = is_correct(answer, gold)

        records.append({
            "problem_id":          prob_id,
            "cap":                 cap,
            "n_tokens":            n_tokens,
            "naturally_converged": naturally_converged,
            "forced":              was_forced,
            "correct":             correct,
            "pred":                answer,
            "gold":                gold,
        })
        print(
            f"  [{i+1:3d}/{N_PROBLEMS}] nat={naturally_converged} forced={was_forced} "
            f"tok={n_tokens:5d} correct={correct} pred={answer} gold={gold}"
        )

    return records


def summarise(cap: int, records, total_tokens_uncapped: int) -> dict:
    n                = len(records)
    n_nat            = sum(r["naturally_converged"] for r in records)
    n_forced         = sum(r["forced"]              for r in records)
    n_correct        = sum(r["correct"]             for r in records)
    n_correct_nat    = sum(r["correct"] for r in records if r["naturally_converged"])
    n_correct_forced = sum(r["correct"] for r in records if r["forced"])
    tokens_used      = sum(r["n_tokens"]            for r in records)
    compute_saved    = (total_tokens_uncapped - tokens_used) / total_tokens_uncapped

    print(
        f"\n  Cap {cap}: acc={n_correct/n:.3f} "
        f"nat_conv={n_nat} forced={n_forced} "
        f"compute_saved={compute_saved:.3f} tokens_used={tokens_used:,}"
    )
    return {
        "cap":                   cap,
        "n_problems":            n,
        "accuracy":              n_correct / n,
        "n_correct":             n_correct,
        "n_naturally_converged": n_nat,
        "n_forced":              n_forced,
        "accuracy_nat_conv":     n_correct_nat  / max(n_nat,    1),
        "accuracy_forced":       n_correct_forced / max(n_forced, 1),
        "tokens_used":           tokens_used,
        "compute_saved_rate":    compute_saved,
    }


def load_uncapped_token_total() -> int:
    """
    Load total tokens from the original uncapped run.
    Falls back to the known value from budget_sweep_extended.json.
    """
    sweep_file = RESULTS_DIR / "budget_sweep_extended.json"
    if sweep_file.exists():
        with open(sweep_file) as f:
            d = json.load(f)
        # sum tokens at the ceiling budget (10000)
        for row in d.get("by_budget", []):
            if row.get("budget") == 10000:
                total = int(row["mean_tokens_generated"] * d["n_problems"])
                print(f"Uncapped total tokens (from sweep): {total:,}")
                return total
    # fallback: value from paper (Table 6 note)
    fallback = 1_414_412
    print(f"Uncapped total tokens (hardcoded fallback): {fallback:,}")
    return fallback


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model, think_end_id = load_model()
    problems                        = load_problems()
    total_tokens_uncapped           = load_uncapped_token_total()

    print(f"\nLoaded {len(problems)} problems | device={device}")
    print(f"Caps to run: {CAPS}\n")

    all_records = []
    summaries   = []

    for cap in CAPS:
        records = run_one(cap, problems, tokenizer, model, think_end_id, device)
        summary = summarise(cap, records, total_tokens_uncapped)
        all_records.extend(records)
        summaries.append(summary)

        # checkpoint after each cap
        with open(OUT_FILE, "w") as f:
            json.dump(
                {
                    "model":                 MODEL_NAME,
                    "n_problems":            N_PROBLEMS,
                    "seed":                  SEED,
                    "caps":                  CAPS,
                    "total_tokens_uncapped": total_tokens_uncapped,
                    "by_cap":                summaries,
                    "results":               all_records,
                },
                f, indent=2,
            )
        print(f"  [checkpoint saved → {OUT_FILE}]")

    # Final table
    print(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}")
    print(f"{'Cap':>6}  {'Acc':>6}  {'Saved%':>7}  {'NatConv':>8}  {'Forced':>7}  {'AccForced':>10}")
    for s in summaries:
        print(
            f"  {s['cap']:4d}  {s['accuracy']:6.3f}  "
            f"{s['compute_saved_rate']*100:6.1f}%  "
            f"{s['n_naturally_converged']:8d}  "
            f"{s['n_forced']:7d}  "
            f"{s['accuracy_forced']:10.3f}"
        )
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
