"""
Contamination check: run uncapped inference on AIME 2025 problems.
These problems were released February 2025, AFTER the model's July 2024
training cutoff — so the model cannot have memorized them.

Compare accuracy here against the 1983-2024 results to assess memorization.

Usage:
  python contamination_check.py

Runtime: ~30 min on a 24GB GPU (30 problems × ~1 min each).
"""

import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_NEW_TOKENS = 10000
THINK_END_ID = 151649
_NETWORK_VOLUME = Path("/runpod-volume")
if _NETWORK_VOLUME.exists():
    RESULTS_DIR     = _NETWORK_VOLUME / "early_detection" / "results"
    CHECKPOINT_FILE = _NETWORK_VOLUME / "early_detection" / "checkpoints" / "contamination_check.json"
    print(f"[INFO] Using network volume: {_NETWORK_VOLUME}")
else:
    RESULTS_DIR     = Path("results")
    CHECKPOINT_FILE = Path("checkpoints/contamination_check.json")


def extract_answer(response):
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    if boxed:
        return boxed[-1].strip()
    ans = re.findall(r"[Tt]he answer is\s*\$?([0-9\-\/\.\,]+)", response)
    if ans:
        return ans[-1].strip()
    post_think = response.split("</think>")[-1] if "</think>" in response else response
    numbers = re.findall(r"\$?([0-9]+(?:\.[0-9]+)?)", post_think)
    if numbers:
        return numbers[-1].strip()
    return None


def normalize_answer(ans):
    if ans is None:
        return None
    ans = ans.replace(",", "").replace("$", "").strip()
    try:
        return str(float(ans))
    except Exception:
        return ans.lower().strip()


def load_aime2025():
    """Try multiple AIME 2025 dataset sources."""
    sources = [
        ("MathArena/aime_2025", "train"),
        ("opencompass/AIME2025", "train"),
        ("yentinglin/aime_2025", "train"),
    ]
    for name, split in sources:
        try:
            ds = load_dataset(name, split=split)
            print(f"Loaded {len(ds)} problems from {name}")
            print(f"Columns: {ds.column_names}")
            # Normalize column names across datasets
            samples = []
            for r in ds:
                q = r.get("problem") or r.get("Question") or r.get("question") or r.get("Problem")
                a = r.get("answer") or r.get("Answer") or r.get("solution")
                if q and a:
                    samples.append({"question": str(q), "answer": str(a)})
            if samples:
                print(f"Extracted {len(samples)} usable samples")
                return samples
        except Exception as e:
            print(f"  Could not load {name}: {e}")
    raise RuntimeError("Could not load any AIME 2025 dataset. Try: pip install datasets --upgrade")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    # Load existing progress
    results = []
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            results = json.load(f)
        print(f"Resuming from sample {len(results)}")

    # Load dataset
    print("Loading AIME 2025 dataset...")
    samples = load_aime2025()

    if len(results) >= len(samples):
        print("All samples already completed!")
    else:
        # Load model
        print(f"\nLoading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
        )
        model.eval()
        print(f"VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

        start_idx = len(results)
        for i, sample in enumerate(samples[start_idx:], start=start_idx):
            t0 = time.time()

            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": sample["question"]}],
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            generated = output[0][prompt_len:]
            decoded = tokenizer.decode(generated, skip_special_tokens=False)

            think_end_positions = (generated == THINK_END_ID).nonzero()
            converged = len(think_end_positions) > 0

            extracted = extract_answer(decoded)
            correct = normalize_answer(extracted) == normalize_answer(sample["answer"])
            elapsed = time.time() - t0

            results.append({
                "idx": i,
                "question_preview": sample["question"][:80],
                "expected": sample["answer"],
                "extracted": extracted,
                "correct": correct,
                "converged": converged,
                "total_tokens": len(generated),
                "elapsed": round(elapsed, 1),
            })

            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(results, f, indent=2)

            del output, generated, inputs
            torch.cuda.empty_cache()

            done = i + 1
            n_conv = sum(1 for r in results if r["converged"])
            n_correct = sum(1 for r in results if r["correct"])
            print(
                f"[{done}/{len(samples)}] "
                f"conv={n_conv}/{done} ({n_conv/done:.1%}) | "
                f"correct={n_correct}/{done} ({n_correct/done:.1%}) | "
                f"{elapsed:.0f}s"
            )

    # ── Final Report ───────────────────────────────────────────────────────────
    n = len(results)
    n_conv = sum(1 for r in results if r["converged"])
    n_correct = sum(1 for r in results if r["correct"])
    n_nonconv = n - n_conv
    conv_correct = sum(1 for r in results if r["converged"] and r["correct"])
    nonconv_correct = sum(1 for r in results if not r["converged"] and r["correct"])

    print("\n" + "=" * 60)
    print("CONTAMINATION CHECK RESULTS — AIME 2025 (post-cutoff)")
    print("=" * 60)
    print(f"  N = {n}")
    print(f"  Convergence rate:  {n_conv}/{n} ({n_conv/n:.1%})")
    print(f"  Overall accuracy:  {n_correct}/{n} ({n_correct/n:.1%})")
    if n_conv > 0:
        print(f"  Converged acc:     {conv_correct}/{n_conv} ({conv_correct/n_conv:.1%})")
    if n_nonconv > 0:
        print(f"  Non-conv acc:      {nonconv_correct}/{n_nonconv} ({nonconv_correct/n_nonconv:.1%})")

    print("\n  COMPARISON vs AIME 1983-2024 (original run):")
    print(f"  {'Metric':<30} {'1983-2024':>12} {'2025':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Convergence rate':<30} {'62.0%':>12} {f'{n_conv/n:.1%}':>12}")
    print(f"  {'Overall accuracy':<30} {'58.5%':>12} {f'{n_correct/n:.1%}':>12}")
    conv_acc_2025 = f"{conv_correct/n_conv:.1%}" if n_conv > 0 else "N/A"
    nonconv_acc_2025 = f"{nonconv_correct/n_nonconv:.1%}" if n_nonconv > 0 else "N/A"
    print(f"  {'Converged accuracy':<30} {'90.3%':>12} {conv_acc_2025:>12}")
    print(f"  {'Non-converged accuracy':<30} {'6.6%':>12} {nonconv_acc_2025:>12}")

    print("\n  INTERPRETATION:")
    overall_2025 = n_correct / n
    if overall_2025 < 0.30:
        print("  ⚠ Large accuracy drop on post-cutoff problems.")
        print("    The 1983-2024 accuracy was likely inflated by memorization.")
        print("    Acknowledge this strongly in Limitations.")
    elif overall_2025 < 0.45:
        print("  ~ Moderate accuracy drop on post-cutoff problems.")
        print("    Partial memorization likely. Acknowledge in Limitations.")
    else:
        print("  ✓ Accuracy on post-cutoff problems is comparable.")
        print("    Memorization is unlikely to be the primary driver of results.")

    summary = {
        "dataset": "AIME 2025 (post-training-cutoff)",
        "n": n,
        "convergence_rate": round(n_conv / n, 4),
        "overall_accuracy": round(n_correct / n, 4),
        "converged_accuracy": round(conv_correct / n_conv, 4) if n_conv > 0 else None,
        "non_converged_accuracy": round(nonconv_correct / n_nonconv, 4) if n_nonconv > 0 else None,
        "comparison_1983_2024": {
            "convergence_rate": 0.62,
            "overall_accuracy": 0.585,
            "converged_accuracy": 0.903,
            "non_converged_accuracy": 0.066,
        },
    }
    out = RESULTS_DIR / "contamination_check_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Full results saved to {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
