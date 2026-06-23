"""
Instrumented generation run for Early Detection project.
Captures early activations + computes behavioral features at checkpoint positions.
Checkpoints after EVERY sample to minimize lost work on crash.

Usage:
  python generate.py                          # default checkpoint at 150 tokens
  python generate.py --checkpoints 100,150,200,300   # sweep multiple positions
"""

import argparse
import json
import math
import time
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SEED = 42
N_SAMPLES = 200
MAX_NEW_TOKENS = 10000
THINK_START_ID = 151648
THINK_END_ID = 151649

RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")


# ── BudgetForcingProcessor (not used for forcing, kept for reference) ────────
# We run uncapped, so no budget forcing needed. But we track thinking state
# to know when </think> is emitted.


# ── Activation & logit capture during generate() ─────────────────────────────
class GenerationInstrumenter:
    """
    Hooks into model during generate() to capture:
    - Hidden state at specific checkpoint token positions
    - Per-step logit entropy for behavioral baseline
    - Token IDs for repetition analysis

    Key design: we compute entropy ON THE FLY from logits rather than storing
    raw logit tensors (which would be ~44MB per step in fp16 for vocab=151k).
    """

    def __init__(self, model, layer_idx, checkpoint_positions, prompt_len):
        self.model = model
        self.layer_idx = layer_idx
        self.checkpoint_positions = sorted(checkpoint_positions)
        self.max_checkpoint = max(checkpoint_positions)
        self.prompt_len = prompt_len

        # Step counter: tracks generated tokens (0-indexed, AFTER prefill)
        # The first forward pass in generate() is prefill (all prompt tokens).
        # We detect it by checking seq_len > 1 and skip it.
        self.gen_step = 0
        self.prefill_done = False

        # Storage for captured data
        self.hidden_states = {}  # {pos: tensor on CPU}
        self.step_entropies = []  # entropy at each generation step
        self.generated_token_ids = []  # for repetition analysis

        # Hooks
        self._hidden_handle = None
        self._logit_handle = None
        self._active = False

    def _hidden_hook(self, module, input, output):
        if not self._active or not self.prefill_done:
            return
        if self.gen_step in self.checkpoint_positions:
            self.hidden_states[self.gen_step] = output[0][:, -1, :].detach().cpu().float()

    def _logit_hook(self, module, input, output):
        if not self._active:
            return

        # Detect prefill: first forward pass has seq_len > 1
        if not self.prefill_done:
            self.prefill_done = True
            return

        # Now gen_step 0 = first generated token
        if self.gen_step <= self.max_checkpoint:
            with torch.no_grad():
                logits = output[0, -1, :].float()
                probs = torch.softmax(logits, dim=0)
                log_probs = torch.log_softmax(logits, dim=0)
                entropy = -(probs * log_probs).sum().item()
                if math.isnan(entropy) or math.isinf(entropy):
                    entropy = 0.0
                self.step_entropies.append(entropy)
                token_id = logits.argmax().item()
                self.generated_token_ids.append(token_id)

        self.gen_step += 1

    def attach(self):
        self._hidden_handle = self.model.model.layers[self.layer_idx].register_forward_hook(
            self._hidden_hook
        )
        self._logit_handle = self.model.lm_head.register_forward_hook(self._logit_hook)
        self._active = True

    def detach(self):
        self._active = False
        if self._hidden_handle:
            self._hidden_handle.remove()
        if self._logit_handle:
            self._logit_handle.remove()

    def get_results(self):
        return {
            "hidden_states": self.hidden_states,
            "step_entropies": self.step_entropies,
            "generated_token_ids": self.generated_token_ids,
        }


# ── Behavioral feature extraction ────────────────────────────────────────────
def compute_behavioral_features(step_entropies, generated_token_ids, checkpoint_pos):
    """
    Compute behavioral-only features at a given checkpoint position.
    These do NOT use internal activations — only surface-level signals.
    """
    ent = step_entropies[:checkpoint_pos]
    tokens = generated_token_ids[:checkpoint_pos]

    if len(ent) == 0:
        return {
            "entropy_mean": 0.0,
            "entropy_max": 0.0,
            "entropy_std": 0.0,
            "entropy_trend": 0.0,
            "repetition_bigram": 0.0,
            "repetition_trigram": 0.0,
            "token_count": 0,
        }

    ent_arr = np.array(ent, dtype=np.float64)

    # Entropy trend: slope of linear fit
    if len(ent_arr) > 1:
        x = np.arange(len(ent_arr))
        slope = np.polyfit(x, ent_arr, 1)[0]
    else:
        slope = 0.0

    # Repetition: fraction of repeated n-grams
    def ngram_repetition(toks, n):
        if len(toks) < n:
            return 0.0
        ngrams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        if not ngrams:
            return 0.0
        return 1.0 - len(set(ngrams)) / len(ngrams)

    return {
        "entropy_mean": float(np.mean(ent_arr)),
        "entropy_max": float(np.max(ent_arr)),
        "entropy_std": float(np.std(ent_arr)),
        "entropy_trend": float(slope),
        "repetition_bigram": ngram_repetition(tokens, 2),
        "repetition_trigram": ngram_repetition(tokens, 3),
        "token_count": len(tokens),
    }


# ── Answer extraction (reused from original project) ─────────────────────────
import re

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


# ── Checkpoint I/O ────────────────────────────────────────────────────────────
def get_checkpoint_path(checkpoint_pos):
    return CHECKPOINT_DIR / f"generation_cp{checkpoint_pos}.json"


def get_activation_path(checkpoint_pos):
    return CHECKPOINT_DIR / f"activations_cp{checkpoint_pos}.pt"


def load_checkpoint(checkpoint_pos):
    """Load both the JSON metadata and the activation tensors."""
    json_path = get_checkpoint_path(checkpoint_pos)
    act_path = get_activation_path(checkpoint_pos)

    records = []
    activations = {}

    if json_path.exists():
        with open(json_path) as f:
            records = json.load(f)

    if act_path.exists():
        activations = torch.load(act_path, map_location="cpu", weights_only=True)

    return records, activations


def save_checkpoint(checkpoint_pos, records, activations):
    """Save JSON metadata and activation tensors separately."""
    json_path = get_checkpoint_path(checkpoint_pos)
    act_path = get_activation_path(checkpoint_pos)

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    torch.save(activations, act_path)


# ── Main generation loop ─────────────────────────────────────────────────────
def run_generation(checkpoint_positions):
    RESULTS_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    print(f"Checkpoint positions: {checkpoint_positions}")

    # Load model
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    print(f"VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    # Determine hook layer
    n_layers = model.config.num_hidden_layers
    layer_idx = int(n_layers * 0.6)
    print(f"Hook layer: {layer_idx} / {n_layers}")

    # Load dataset
    print("Loading AIME dataset...")
    ds = load_dataset("gneubig/aime-1983-2024", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(N_SAMPLES, len(ds))))
    samples = [{"question": r["Question"], "answer": str(r["Answer"])} for r in ds]
    print(f"Loaded {len(samples)} samples")

    # For each checkpoint position, load existing progress
    # We use the FIRST checkpoint position as the primary — all positions
    # are captured in the same generation run, so they share progress.
    primary_cp = checkpoint_positions[0]
    existing_records, existing_activations = load_checkpoint(primary_cp)
    start_idx = len(existing_records)

    # Also load other checkpoint positions' existing data
    all_records = {cp: [] for cp in checkpoint_positions}
    all_activations = {cp: {} for cp in checkpoint_positions}
    for cp in checkpoint_positions:
        recs, acts = load_checkpoint(cp)
        all_records[cp] = recs
        all_activations[cp] = acts

    # Verify all checkpoint positions have the same progress
    lengths = {cp: len(all_records[cp]) for cp in checkpoint_positions}
    start_idx = min(lengths.values())
    if max(lengths.values()) != start_idx:
        print(f"WARNING: checkpoint positions have different progress: {lengths}")
        print(f"Resuming from the minimum: {start_idx}")
        for cp in checkpoint_positions:
            all_records[cp] = all_records[cp][:start_idx]
            keys_to_remove = [k for k in all_activations[cp] if k >= start_idx]
            for k in keys_to_remove:
                del all_activations[cp][k]

    if start_idx > 0:
        print(f"Resuming from sample {start_idx}/{len(samples)}")
    if start_idx >= len(samples):
        print("All samples already completed!")
        return

    # Generation loop
    total_time = 0.0
    for i in range(start_idx, len(samples)):
        sample = samples[i]
        t0 = time.time()

        # Prepare prompt
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["question"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Set up instrumenter
        instrumenter = GenerationInstrumenter(
            model, layer_idx, checkpoint_positions, prompt_len
        )
        instrumenter.attach()

        # Run generation
        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        instrumenter.detach()
        instr_results = instrumenter.get_results()

        # Decode output
        generated = output[0][prompt_len:]
        decoded = tokenizer.decode(generated, skip_special_tokens=False)

        # Determine convergence
        think_end_positions = (generated == THINK_END_ID).nonzero()
        converged = len(think_end_positions) > 0
        think_end_pos = think_end_positions[0].item() if converged else None

        # Determine correctness
        extracted = extract_answer(decoded)
        correct = normalize_answer(extracted) == normalize_answer(sample["answer"])

        total_tokens = len(generated)
        elapsed = time.time() - t0
        total_time += elapsed

        # Save per-checkpoint data
        for cp in checkpoint_positions:
            behavioral = compute_behavioral_features(
                instr_results["step_entropies"],
                instr_results["generated_token_ids"],
                cp,
            )

            record = {
                "idx": i,
                "converged": converged,
                "correct": correct,
                "total_tokens": total_tokens,
                "think_end_pos": think_end_pos,
                "hit_max_tokens": total_tokens >= MAX_NEW_TOKENS,
                "extracted_answer": extracted,
                "expected_answer": sample["answer"],
                "behavioral_features": behavioral,
                "elapsed": round(elapsed, 1),
            }
            all_records[cp].append(record)

            # Save activation if captured
            if cp in instr_results["hidden_states"]:
                all_activations[cp][i] = instr_results["hidden_states"][cp]
            else:
                # Checkpoint position beyond generation length
                all_activations[cp][i] = None

            # Checkpoint after EVERY sample
            save_checkpoint(cp, all_records[cp], all_activations[cp])

        # Free GPU memory
        del output, generated, inputs, instrumenter
        torch.cuda.empty_cache()

        # Progress report
        conv_so_far = sum(1 for r in all_records[primary_cp] if r["converged"])
        done = i + 1
        avg_time = total_time / (done - start_idx)
        remaining = (len(samples) - done) * avg_time
        remaining_h = remaining / 3600

        print(
            f"[{done}/{len(samples)}] "
            f"conv={conv_so_far}/{done} ({conv_so_far / done:.1%}) | "
            f"correct={sum(1 for r in all_records[primary_cp] if r['correct'])}/{done} | "
            f"tokens={total_tokens} | "
            f"{elapsed:.0f}s | "
            f"ETA: {remaining_h:.1f}h"
        )

    # Final summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    for cp in checkpoint_positions:
        recs = all_records[cp]
        n_conv = sum(1 for r in recs if r["converged"])
        n_correct = sum(1 for r in recs if r["correct"])
        conv_correct = sum(1 for r in recs if r["converged"] and r["correct"])
        nonconv_correct = sum(1 for r in recs if not r["converged"] and r["correct"])
        n_nonconv = len(recs) - n_conv
        n_with_act = sum(1 for k, v in all_activations[cp].items() if v is not None)
        print(f"\n  Checkpoint {cp} tokens:")
        print(f"    Converged:     {n_conv}/{len(recs)} ({n_conv / len(recs):.1%})")
        print(f"    Accuracy:      {n_correct}/{len(recs)} ({n_correct / len(recs):.1%})")
        print(f"    Conv+Correct:  {conv_correct}/{n_conv} ({conv_correct / n_conv:.1%})" if n_conv else "")
        print(f"    NonC+Correct:  {nonconv_correct}/{n_nonconv} ({nonconv_correct / n_nonconv:.1%})" if n_nonconv else "")
        print(f"    Activations:   {n_with_act}/{len(recs)}")
    print("=" * 60)
    print("\nNext step: python analyze.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="150",
        help="Comma-separated checkpoint positions (tokens into thinking chain)",
    )
    args = parser.parse_args()
    checkpoint_positions = [int(x.strip()) for x in args.checkpoints.split(",")]
    run_generation(checkpoint_positions)


if __name__ == "__main__":
    main()
