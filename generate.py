"""
Instrumented generation for Early Detection project.

Captures two things in one generation run:
  1. ALL 28 layer activations at token 150 → for layer sweep analysis
  2. Layer 16 activations at {50,75,100,125,150,175,200,250,300} → for checkpoint sweep

Checkpoints after EVERY sample. Resume-safe on crash.

Usage:
  python generate.py
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME       = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SEED             = 42
N_SAMPLES        = 200
MAX_NEW_TOKENS   = 10000
THINK_END_ID     = 151649

# Layer sweep: capture ALL layers at this single checkpoint
LAYER_SWEEP_CHECKPOINT = 150

# Checkpoint sweep: capture layer 16 at all these positions
CHECKPOINT_POSITIONS = [50, 75, 100, 125, 150, 175, 200, 250, 300]

RESULTS_DIR    = Path("results")
CHECKPOINT_DIR = Path("checkpoints")


# ── Instrumenter ──────────────────────────────────────────────────────────────
class GenerationInstrumenter:
    """
    Attaches hooks during model.generate() to capture:

    - Hidden states from ALL layers at LAYER_SWEEP_CHECKPOINT (for layer sweep)
    - Hidden state from layer 16 at every position in CHECKPOINT_POSITIONS (for checkpoint sweep)
    - Per-step entropy and token IDs up to max(CHECKPOINT_POSITIONS) (for behavioral baseline)

    Prefill detection: the first forward pass processes the entire prompt
    (seq_len > 1). We skip it so gen_step=0 is the first generated token.

    Memory: activations are moved to CPU immediately. We never hold >1 step
    of GPU activations at a time.
    """

    def __init__(self, model, n_layers, layer_sweep_checkpoint, checkpoint_positions):
        self.model = model
        self.n_layers = n_layers
        self.layer_sweep_checkpoint = layer_sweep_checkpoint
        self.checkpoint_positions = set(checkpoint_positions)
        self.max_checkpoint = max(checkpoint_positions)
        self.sweep_layer_idx = int(n_layers * 0.6)  # layer 16 for 28-layer model

        self.gen_step = 0
        self.prefill_done = False

        # layer sweep: {layer_idx: tensor [1, hidden]}
        self.layer_sweep_activations = {}

        # checkpoint sweep: {checkpoint_pos: tensor [1, hidden]}
        self.checkpoint_activations = {}

        # behavioral: entropy and token id at each step
        self.step_entropies = []
        self.generated_token_ids = []

        self._handles = []

    def _make_layer_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if not self.prefill_done:
                return
            if self.gen_step == self.layer_sweep_checkpoint:
                h = output[0].detach().cpu().float()
                if h.dim() == 3:
                    h = h[:, -1, :]
                self.layer_sweep_activations[layer_idx] = h
        return hook_fn

    def _sweep_layer_hook(self, module, input, output):
        """Hook on layer 16 only — captures at every checkpoint position."""
        if not self.prefill_done:
            return
        if self.gen_step in self.checkpoint_positions:
            h = output[0].detach().cpu().float()
            if h.dim() == 3:
                h = h[:, -1, :]
            self.checkpoint_activations[self.gen_step] = h

    def _logit_hook(self, module, input, output):
        if not self.prefill_done:
            self.prefill_done = True
            return

        if self.gen_step <= self.max_checkpoint:
            with torch.no_grad():
                logits = output[0, -1, :].float()
                probs = torch.softmax(logits, dim=0)
                log_probs = torch.log_softmax(logits, dim=0)
                entropy = -(probs * log_probs).sum().item()
                if math.isnan(entropy) or math.isinf(entropy):
                    entropy = 0.0
                self.step_entropies.append(entropy)
                self.generated_token_ids.append(logits.argmax().item())

        self.gen_step += 1

    def attach(self):
        # Hook every layer for the layer sweep
        for i, layer in enumerate(self.model.model.layers):
            if i == self.sweep_layer_idx:
                # This layer doubles as the checkpoint-sweep layer
                def combined_hook(module, input, output, _i=i):
                    if not self.prefill_done:
                        return
                    h = output[0].detach().cpu().float()
                    if h.dim() == 3:
                        h = h[:, -1, :]
                    # Layer sweep
                    if self.gen_step == self.layer_sweep_checkpoint:
                        self.layer_sweep_activations[_i] = h
                    # Checkpoint sweep
                    if self.gen_step in self.checkpoint_positions:
                        self.checkpoint_activations[self.gen_step] = h
                self._handles.append(layer.register_forward_hook(combined_hook))
            else:
                self._handles.append(
                    layer.register_forward_hook(self._make_layer_hook(i))
                )
        # Logit hook
        self._handles.append(
            self.model.lm_head.register_forward_hook(self._logit_hook)
        )

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ── Behavioral features ───────────────────────────────────────────────────────
def compute_behavioral_features(step_entropies, generated_token_ids, checkpoint_pos):
    ent = step_entropies[:checkpoint_pos]
    tokens = generated_token_ids[:checkpoint_pos]

    if len(ent) == 0:
        return {
            "entropy_mean": 0.0, "entropy_max": 0.0,
            "entropy_std": 0.0, "entropy_trend": 0.0,
            "repetition_bigram": 0.0, "repetition_trigram": 0.0,
            "token_count": 0,
        }

    ent_arr = np.array(ent, dtype=np.float64)
    slope = np.polyfit(np.arange(len(ent_arr)), ent_arr, 1)[0] if len(ent_arr) > 1 else 0.0

    def ngram_rep(toks, n):
        if len(toks) < n:
            return 0.0
        ngrams = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
        return 1.0 - len(set(ngrams)) / len(ngrams)

    return {
        "entropy_mean":       float(np.mean(ent_arr)),
        "entropy_max":        float(np.max(ent_arr)),
        "entropy_std":        float(np.std(ent_arr)),
        "entropy_trend":      float(slope),
        "repetition_bigram":  ngram_rep(tokens, 2),
        "repetition_trigram": ngram_rep(tokens, 3),
        "token_count":        len(tokens),
    }


# ── Answer extraction ─────────────────────────────────────────────────────────
def extract_answer(response):
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    if boxed:
        return boxed[-1].strip()
    ans = re.findall(r"[Tt]he answer is\s*\$?([0-9\-\/\.\,]+)", response)
    if ans:
        return ans[-1].strip()
    post = response.split("</think>")[-1] if "</think>" in response else response
    nums = re.findall(r"\$?([0-9]+(?:\.[0-9]+)?)", post)
    return nums[-1].strip() if nums else None


def normalize_answer(ans):
    if ans is None:
        return None
    ans = ans.replace(",", "").replace("$", "").strip()
    try:
        return str(float(ans))
    except Exception:
        return ans.lower().strip()


# ── Checkpoint I/O ────────────────────────────────────────────────────────────
RECORDS_FILE       = CHECKPOINT_DIR / "records.json"
LAYER_SWEEP_FILE   = CHECKPOINT_DIR / "layer_sweep.pt"   # {sample_idx: {layer_idx: tensor}}
CHECKPOINT_ACT_FILE = CHECKPOINT_DIR / "checkpoint_acts.pt"  # {cp_pos: {sample_idx: tensor}}


def load_progress():
    records = []
    layer_sweep = {}
    checkpoint_acts = {cp: {} for cp in CHECKPOINT_POSITIONS}

    if RECORDS_FILE.exists():
        with open(RECORDS_FILE) as f:
            records = json.load(f)
    if LAYER_SWEEP_FILE.exists():
        layer_sweep = torch.load(LAYER_SWEEP_FILE, map_location="cpu", weights_only=True)
    if CHECKPOINT_ACT_FILE.exists():
        saved = torch.load(CHECKPOINT_ACT_FILE, map_location="cpu", weights_only=True)
        for cp in CHECKPOINT_POSITIONS:
            checkpoint_acts[cp] = saved.get(cp, {})

    return records, layer_sweep, checkpoint_acts


def save_progress(records, layer_sweep, checkpoint_acts):
    with open(RECORDS_FILE, "w") as f:
        json.dump(records, f, indent=2)
    torch.save(layer_sweep, LAYER_SWEEP_FILE)
    torch.save(checkpoint_acts, CHECKPOINT_ACT_FILE)


# ── Main generation loop ──────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    print(f"Layer sweep at:       token {LAYER_SWEEP_CHECKPOINT} (all {28} layers)")
    print(f"Checkpoint sweep at:  {CHECKPOINT_POSITIONS} (layer ~60% depth)")

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    sweep_layer = int(n_layers * 0.6)
    print(f"VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    print(f"Layers: {n_layers} | Sweep layer (60% depth): {sweep_layer}")

    # Load dataset
    print("\nLoading AIME dataset...")
    ds = load_dataset("gneubig/aime-1983-2024", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(N_SAMPLES, len(ds))))
    samples = [{"question": r["Question"], "answer": str(r["Answer"])} for r in ds]
    print(f"Loaded {len(samples)} samples")

    # Load existing progress
    records, layer_sweep, checkpoint_acts = load_progress()
    start_idx = len(records)
    if start_idx > 0:
        print(f"Resuming from sample {start_idx}/{len(samples)}")
    if start_idx >= len(samples):
        print("All samples completed!")
        return

    # Generation loop
    total_time = 0.0
    for i in range(start_idx, len(samples)):
        sample = samples[i]
        t0 = time.time()

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["question"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        instrumenter = GenerationInstrumenter(
            model, n_layers, LAYER_SWEEP_CHECKPOINT, CHECKPOINT_POSITIONS
        )
        instrumenter.attach()

        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        instrumenter.detach()

        # Decode + labels
        generated = output[0][prompt_len:]
        decoded = tokenizer.decode(generated, skip_special_tokens=False)
        think_end_pos = (generated == THINK_END_ID).nonzero()
        converged = len(think_end_pos) > 0
        extracted = extract_answer(decoded)
        correct = normalize_answer(extracted) == normalize_answer(sample["answer"])
        total_tokens = len(generated)
        elapsed = time.time() - t0
        total_time += elapsed

        # Store layer sweep activations for this sample
        layer_sweep[i] = instrumenter.layer_sweep_activations  # {layer_idx: tensor}

        # Store checkpoint activations
        for cp in CHECKPOINT_POSITIONS:
            act = instrumenter.checkpoint_activations.get(cp)
            checkpoint_acts[cp][i] = act  # None if generation ended before cp

        # Per-checkpoint behavioral features
        cp_behavioral = {
            cp: compute_behavioral_features(
                instrumenter.step_entropies,
                instrumenter.generated_token_ids,
                cp
            )
            for cp in CHECKPOINT_POSITIONS
        }

        records.append({
            "idx": i,
            "converged": converged,
            "correct": correct,
            "total_tokens": total_tokens,
            "think_end_pos": think_end_pos[0].item() if converged else None,
            "hit_max_tokens": total_tokens >= MAX_NEW_TOKENS,
            "extracted_answer": extracted,
            "expected_answer": sample["answer"],
            "behavioral_features": cp_behavioral,
            "elapsed": round(elapsed, 1),
        })

        # Checkpoint after every sample
        save_progress(records, layer_sweep, checkpoint_acts)

        # Cleanup
        del output, generated, inputs, instrumenter
        torch.cuda.empty_cache()

        # Progress
        done = i + 1
        n_conv = sum(1 for r in records if r["converged"])
        n_correct = sum(1 for r in records if r["correct"])
        avg_t = total_time / (done - start_idx)
        eta_h = (len(samples) - done) * avg_t / 3600
        print(
            f"[{done}/{len(samples)}] "
            f"conv={n_conv}/{done} ({n_conv/done:.1%}) | "
            f"correct={n_correct}/{done} ({n_correct/done:.1%}) | "
            f"tokens={total_tokens} | {elapsed:.0f}s | ETA: {eta_h:.1f}h"
        )

    # Final summary
    n = len(records)
    n_conv = sum(1 for r in records if r["converged"])
    n_correct = sum(1 for r in records if r["correct"])
    conv_correct = sum(1 for r in records if r["converged"] and r["correct"])
    nonconv_correct = sum(1 for r in records if not r["converged"] and r["correct"])
    n_nonconv = n - n_conv

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print(f"  Converged:     {n_conv}/{n} ({n_conv/n:.1%})")
    print(f"  Accuracy:      {n_correct}/{n} ({n_correct/n:.1%})")
    if n_conv:
        print(f"  Conv+Correct:  {conv_correct}/{n_conv} ({conv_correct/n_conv:.1%})")
    if n_nonconv:
        print(f"  NonC+Correct:  {nonconv_correct}/{n_nonconv} ({nonconv_correct/n_nonconv:.1%})")
    print(f"  Layer sweep activations: {len(layer_sweep)} samples")
    print("=" * 60)
    print("\nNext step: python analyze.py")


if __name__ == "__main__":
    main()
