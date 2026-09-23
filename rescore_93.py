"""
GPU script: force </think> at 10,000 tokens for the 93 non-converged problems
and score the answers.

Background: the paper currently reports 0.0% accuracy for non-converged
generations. This is a scoring-rule artifact — the pipeline never asks for an
answer when </think> is absent. This script injects </think> and generates up
to 1,000 additional tokens to get an answer, then scores it.

Result determines which abstract form to use:
  If rescored accuracy <= 46.5%:
    "saves about 39% of inference compute with accuracy no lower than
     generating to the 10,000-token ceiling"
  If rescored accuracy > 46.5%:
    "saves about 39% of inference compute at a cost of X points of accuracy"

The 93 non-converged indices come from final_aime_fixed_cap.json (uncapped run).

Usage (on GPU node):
  python rescore_93.py

Outputs:
  /Users/renukaoladri/Downloads/early_detection_25/rescore_93_results.json
"""

import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME        = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATASET_NAME      = "gneubig/aime-1983-2024"
DATASET_SEED      = 42          # must match original generate.py
N_SAMPLES         = 200
MAX_THINK_TOKENS  = 10_000      # generation stops here
MAX_ANSWER_TOKENS = 1_000       # tokens to generate after injecting </think>

FIXED_CAP_PATH = Path(__file__).parent / "final_aime_fixed_cap.json"
OUT_PATH       = Path("/Users/renukaoladri/Downloads/early_detection_25/rescore_93_results.json")

# Token ID for </think> in DeepSeek-R1
THINK_END_ID   = 151649


# ── Answer extraction / scoring ────────────────────────────────────────────────
def extract_answer(text):
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    ans = re.findall(r"[Tt]he answer is\s*\$?([0-9\-\/\.\,]+)", text)
    if ans:
        return ans[-1].strip()
    nums = re.findall(r"\$?([0-9]+(?:\.[0-9]+)?)", text.split("</think>")[-1])
    return nums[-1].strip() if nums else None


def normalize_answer(ans):
    if ans is None:
        return None
    ans = str(ans).replace(",", "").replace("$", "").strip()
    try:
        return str(float(ans))
    except Exception:
        return ans.lower().strip()


def is_correct(pred, gold):
    return normalize_answer(pred) == normalize_answer(gold)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # ── Load non-converged indices ─────────────────────────────────────────────
    fixed_cap = json.load(open(FIXED_CAP_PATH))
    uncapped_rows = fixed_cap["uncapped"]["rows"]
    nonconv_rows  = [r for r in uncapped_rows if not r.get("converged", True)]
    nonconv_idx   = set(r["idx"] for r in nonconv_rows)
    print(f"Non-converged problems to rescore: {len(nonconv_idx)}")
    assert len(nonconv_idx) == 93, f"Expected 93, got {len(nonconv_idx)}"

    # ── Load dataset (same shuffle as original run) ────────────────────────────
    print(f"\nLoading dataset {DATASET_NAME} (seed={DATASET_SEED})...")
    ds = load_dataset(DATASET_NAME, split="train")
    ds = ds.shuffle(seed=DATASET_SEED).select(range(N_SAMPLES))
    samples = [{"question": r["Question"], "answer": str(r["Answer"])} for r in ds]
    print(f"Dataset: {len(samples)} samples loaded")

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    # ── Think-end token ────────────────────────────────────────────────────────
    # Verify the </think> token id
    think_end_str = "</think>"
    think_end_ids = tokenizer.encode(think_end_str, add_special_tokens=False)
    print(f"</think> encodes to token IDs: {think_end_ids}")
    # Use the canonical id — for DeepSeek-R1 it's 151649
    think_end_token = THINK_END_ID

    # ── Injection string: </think>\n\n ─────────────────────────────────────────
    # After 10k tokens with no </think>, we inject this and ask model to answer.
    inject_text  = "\n\n</think>\n\nThe answer is $\\boxed{"
    inject_ids   = tokenizer.encode(inject_text, add_special_tokens=False)
    inject_tensor = torch.tensor([inject_ids], device="cuda")

    # ── Rescore loop ───────────────────────────────────────────────────────────
    results = []
    n_correct   = 0
    n_no_answer = 0

    for i in sorted(nonconv_idx):
        sample = samples[i]
        t0 = time.time()

        # Build prompt
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["question"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda")

        # Step 1: generate up to MAX_THINK_TOKENS
        with torch.no_grad():
            think_output = model.generate(
                prompt_ids,
                max_new_tokens=MAX_THINK_TOKENS,
                do_sample=False,
                temperature=1.0,
                eos_token_id=think_end_token,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_think = think_output[0][prompt_ids.shape[1]:]
        already_has_think = (think_end_token in generated_think.tolist())

        if already_has_think:
            # Converged naturally — decode full response and score
            full_text = tokenizer.decode(think_output[0], skip_special_tokens=True)
            extracted = extract_answer(full_text)
            forced    = False
        else:
            # Non-converged: inject </think> and generate answer
            # Concatenate: prompt + think tokens + injection string
            combined = torch.cat([think_output, inject_tensor], dim=1)

            with torch.no_grad():
                answer_output = model.generate(
                    combined,
                    max_new_tokens=MAX_ANSWER_TOKENS,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode only the answer portion
            answer_tokens = answer_output[0][combined.shape[1]:]
            answer_text   = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            # Also try decoding the full output for extract_answer to work on
            full_text = inject_text + answer_text
            extracted = extract_answer(full_text)
            forced    = True

        correct = is_correct(extracted, sample["answer"])
        if correct:
            n_correct += 1
        if extracted is None:
            n_no_answer += 1

        elapsed = time.time() - t0
        print(
            f"[{len(results)+1:3d}/93] idx={i:3d} "
            f"forced={forced} correct={correct} "
            f"pred={extracted!r} gold={sample['answer']!r} "
            f"({elapsed:.1f}s)",
            flush=True
        )

        results.append({
            "idx":          i,
            "forced":       forced,
            "correct":      correct,
            "extracted":    extracted,
            "gold":         sample["answer"],
            "question_len": len(sample["question"]),
        })

    # ── Summary ────────────────────────────────────────────────────────────────
    n = len(results)
    acc = n_correct / n
    print(f"\n{'='*50}")
    print(f"Rescored {n} non-converged problems")
    print(f"  Correct:    {n_correct}/{n} = {acc:.1%}")
    print(f"  No answer:  {n_no_answer}/{n}")
    print(f"\n  If acc <= 46.5%: use 'no lower than' form in abstract")
    print(f"  If acc >  46.5%: use 'at a cost of {acc:.1%} accuracy' form")

    output = {
        "description": "Forced </think> rescoring of 93 non-converged seed-0 AIME problems",
        "n_problems":  n,
        "n_correct":   n_correct,
        "accuracy":    round(acc, 6),
        "n_no_answer": n_no_answer,
        "method": (
            f"Generate up to {MAX_THINK_TOKENS} tokens; if no </think>, inject "
            f"'\\n\\n</think>\\n\\nThe answer is $\\boxed{{' and generate "
            f"up to {MAX_ANSWER_TOKENS} more tokens."
        ),
        "abstract_form": (
            "no lower than" if acc <= 0.465 else f"at a cost of {acc:.1%} accuracy"
        ),
        "results": results,
    }
    json.dump(output, open(OUT_PATH, "w"), indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
