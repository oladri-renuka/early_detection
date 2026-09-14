"""
Run two additional independent greedy runs on the same 200 AIME problems,
then compute item-level pairwise agreement across all three greedy runs.

Run 1 already exists in checkpoints/records.json (convergence 57.5%).
This script produces Runs 2 and 3, then reports:
  - Pairwise item-level agreement (runs 1-2, 1-3, 2-3)
  - Three-way stability counts
  - Flip rate within greedy decoding (control for the temperature flip rate)

Outputs: results/greedy_item_diff.json

Usage: python greedy_runs_item_diff.py
"""

import json
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
N_PROBLEMS  = 200
SEED        = 0
MAX_TOKENS  = 10_000
NEW_RUNS    = [2, 3]   # run indices to generate (run 1 = records.json)

_NETWORK_VOLUME = Path("/runpod-volume")
if _NETWORK_VOLUME.exists():
    RESULTS_DIR    = _NETWORK_VOLUME / "early_detection" / "results"
    CHECKPOINT_DIR = _NETWORK_VOLUME / "early_detection" / "checkpoints"
else:
    RESULTS_DIR    = Path("results")
    CHECKPOINT_DIR = Path("checkpoints")

OUT_FILE = RESULTS_DIR / "greedy_item_diff.json"


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
        MODEL_NAME, dtype=torch.float16, device_map="auto"
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return tokenizer, model


def load_problems():
    ds  = load_dataset("gneubig/aime-1983-2024", split="train")
    rng = random.Random(SEED)
    return rng.sample(list(ds), N_PROBLEMS)


def run_greedy(run_idx, problems, tokenizer, model, device):
    print(f"\n{'='*50}\nGreedy Run {run_idx} (uncapped, no LogitsProcessor)\n{'='*50}")
    results = []
    for i, item in enumerate(problems):
        gold   = str(item["Answer"])
        prob_id = item.get("ID", i)
        messages = [{"role": "user", "content": item["Question"]}]
        prompt = (
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            + "<think>\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        text      = tokenizer.decode(generated, skip_special_tokens=False)
        n_tokens  = len(generated)
        converged = "</think>" in text
        answer    = extract_answer(text)
        correct   = is_correct(answer, gold)
        results.append({
            "idx":       i,
            "problem_id": prob_id,
            "converged": converged,
            "correct":   correct,
            "n_tokens":  n_tokens,
        })
        if (i + 1) % 20 == 0:
            n_conv = sum(r["converged"] for r in results)
            print(f"  [{i+1:3d}/{N_PROBLEMS}] conv_rate={n_conv/(i+1):.3f}")

    n_conv = sum(r["converged"] for r in results)
    print(f"  Run {run_idx}: convergence={n_conv}/{N_PROBLEMS} = {n_conv/N_PROBLEMS:.3f}")
    return results


def load_run1():
    with open(CHECKPOINT_DIR / "records.json") as f:
        records = json.load(f)
    return {r["idx"]: r["converged"] for r in records}


def pairwise_agreement(labels_a: dict, labels_b: dict, name: str) -> dict:
    common = sorted(set(labels_a) & set(labels_b))
    agree  = sum(labels_a[i] == labels_b[i] for i in common)
    flips  = len(common) - agree
    return {
        "comparison": name,
        "n_common":   len(common),
        "agreement":  agree / len(common),
        "flip_rate":  flips / len(common),
        "n_flips":    flips,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model = load_model()
    problems         = load_problems()
    print(f"Loaded {len(problems)} problems")

    # Load existing run 1
    run1_labels = load_run1()
    print(f"Run 1 (from records.json): convergence={sum(run1_labels.values())}/{len(run1_labels)}")

    all_runs = {1: run1_labels}
    raw_runs = {}

    for run_idx in NEW_RUNS:
        results = run_greedy(run_idx, problems, tokenizer, model, device)
        raw_runs[run_idx] = results
        all_runs[run_idx] = {r["idx"]: r["converged"] for r in results}

        # Checkpoint
        with open(OUT_FILE, "w") as f:
            json.dump({"raw_runs": {str(k): v for k, v in raw_runs.items()}}, f, indent=2)

    # Pairwise agreement
    pairs = [
        (1, 2, "run1_vs_run2"),
        (1, 3, "run1_vs_run3"),
        (2, 3, "run2_vs_run3"),
    ]
    pairwise = [pairwise_agreement(all_runs[a], all_runs[b], name) for a, b, name in pairs]

    # Three-way stability
    idxs = sorted(set(all_runs[1]) & set(all_runs[2]) & set(all_runs[3]))
    stable_conv     = sum(all(all_runs[r][i] for r in [1,2,3]) for i in idxs)
    stable_nonconv  = sum(not any(all_runs[r][i] for r in [1,2,3]) for i in idxs)
    mixed           = len(idxs) - stable_conv - stable_nonconv
    mean_flip_rate  = sum(p["flip_rate"] for p in pairwise) / len(pairwise)

    print(f"\n{'='*50}\nPAIRWISE GREEDY AGREEMENT\n{'='*50}")
    for p in pairwise:
        print(f"  {p['comparison']}: agreement={p['agreement']:.3f}  flip_rate={p['flip_rate']:.3f}  ({p['n_flips']}/{p['n_common']} flips)")
    print(f"\nThree-way: stable_conv={stable_conv}  stable_nonconv={stable_nonconv}  mixed={mixed}")
    print(f"Mean pairwise flip rate (greedy): {mean_flip_rate:.3f}")
    print(f"Temperature flip rate (from item_level_diff.json): 0.490")
    print(f"\nInterpretation: {mean_flip_rate:.1%} of problems flip between greedy runs vs 49.0% between greedy and temperature.")

    output = {
        "run_convergence_rates": {
            str(k): sum(v.values()) / len(v) for k, v in all_runs.items()
        },
        "pairwise": pairwise,
        "three_way": {
            "n_problems":      len(idxs),
            "stable_converged":    stable_conv,
            "stable_non_converged": stable_nonconv,
            "mixed":           mixed,
        },
        "mean_greedy_flip_rate": mean_flip_rate,
        "temperature_flip_rate": 0.490,
        "raw_runs": {str(k): v for k, v in raw_runs.items()},
    }
    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
