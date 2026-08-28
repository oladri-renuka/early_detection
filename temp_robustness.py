"""
Robustness check: does the bimodal convergence pattern persist at temperature=0.6?
Runs 200 AIME 1983-2024 problems at temperature=0.6 with k=4 seeds.
Each seed sees the same 200 problems (fixed sample, seed-0 draw).
Takes ~8-12h on a single A100.
"""

import argparse
import json, random
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
N_PROBLEMS  = 200
SEEDS       = [0, 1, 2, 3]
TEMPERATURE = 0.6
MAX_TOKENS  = 10000
THINK_END   = "</think>"
OUT_FILE    = Path("temp_robustness_results.json")


def sample_problems(problems):
    """Draw the fixed 200-problem set (always seed-0 so it's reproducible)."""
    rng = random.Random(0)
    return rng.sample(problems, N_PROBLEMS)


def run_seed(seed, problems, tokenizer, model, device):
    torch.manual_seed(seed)
    results = []

    for i, item in enumerate(problems):
        problem = item["Question"]
        messages = [{"role": "user", "content": problem}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + "<think>\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=0.95,
            )

        generated = out[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=False)
        converged = THINK_END in text
        n_tokens = len(generated)

        results.append({
            "problem_idx": item.get("ID", i),
            "converged": converged,
            "n_tokens": n_tokens,
        })
        print(f"seed={seed} [{i+1}/{N_PROBLEMS}] converged={converged} tokens={n_tokens}")

    conv_rate = sum(r["converged"] for r in results) / len(results)
    print(f"\nSeed {seed}: convergence_rate={conv_rate:.3f} ({sum(r['converged'] for r in results)}/{len(results)})")
    return results, conv_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_seed", type=int, default=0,
                        help="Resume from this seed index (skips earlier seeds)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    print("Loading dataset...")
    ds = load_dataset("gneubig/aime-1983-2024", split="train")
    all_problems = list(ds)
    problems = sample_problems(all_problems)
    print(f"Fixed problem set: {len(problems)} problems")

    # Load any existing results so we can append
    per_seed = {}
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            existing = json.load(f)
        per_seed = existing.get("per_seed", {})
        print(f"Loaded existing results for seeds: {list(per_seed.keys())}")

    for seed in SEEDS:
        if seed < args.start_seed:
            print(f"Skipping seed {seed} (--start_seed={args.start_seed})")
            continue
        if f"seed_{seed}" in per_seed:
            print(f"Skipping seed {seed} (already in results file)")
            continue
        print(f"\n{'='*60}\nRunning seed {seed}\n{'='*60}")
        results, conv_rate = run_seed(seed, problems, tokenizer, model, device)
        per_seed[f"seed_{seed}"] = {
            "convergence_rate": conv_rate,
            "n_converged": sum(r["converged"] for r in results),
            "n_total": len(results),
            "results": results,
        }
        # Save incrementally so a crash doesn't lose everything
        _save(per_seed)

    mean_conv = sum(v["convergence_rate"] for v in per_seed.values()) / len(SEEDS)
    out = {
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "n_problems": N_PROBLEMS,
        "seeds": SEEDS,
        "mean_convergence_rate": mean_conv,
        "per_seed": per_seed,
    }
    _save(out)
    print(f"\nDone. Mean convergence rate at temp={TEMPERATURE}: {mean_conv:.3f}")
    print(f"Results saved to {OUT_FILE}")


def _save(data):
    with open(OUT_FILE, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
