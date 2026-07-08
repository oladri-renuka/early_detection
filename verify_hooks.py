"""
Verification script — run BEFORE the expensive generation.
Catches: wrong model, broken hooks, wrong token IDs, bad chat template,
shape mismatches, VRAM issues. Takes < 1 minute.
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
THINK_START_ID = 151648
THINK_END_ID = 151649

def main():
    errors = []

    print("=" * 60)
    print("VERIFICATION: Early Detection Project")
    print("=" * 60)

    # 1. Load model
    print("\n[1/7] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    vram = torch.cuda.memory_allocated(0) / 1e9
    print(f"  Model loaded. VRAM: {vram:.2f} GB")
    if vram > 20:
        errors.append(f"VRAM usage {vram:.1f}GB is dangerously high for 24GB A5000")

    # 2. Verify token IDs
    print("\n[2/7] Verifying token IDs...")
    added = tokenizer.get_added_vocab()
    inv = {v: k for k, v in added.items()}
    start_tok = inv.get(THINK_START_ID, "NOT FOUND")
    end_tok = inv.get(THINK_END_ID, "NOT FOUND")
    print(f"  THINK_START_ID {THINK_START_ID} -> '{start_tok}'")
    print(f"  THINK_END_ID   {THINK_END_ID} -> '{end_tok}'")
    if "<think>" not in start_tok:
        errors.append(f"THINK_START_ID {THINK_START_ID} maps to '{start_tok}', expected '<think>'")
    if "</think>" not in end_tok:
        errors.append(f"THINK_END_ID {THINK_END_ID} maps to '{end_tok}', expected '</think>'")

    # 3. Verify layer count and hook registration
    print("\n[3/7] Verifying model architecture...")
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  num_hidden_layers = {n_layers}")
    print(f"  hidden_size = {hidden_dim}")
    print(f"  Will hook all {n_layers} layers for layer sweep")
    print(f"  Sweep layer (60% depth) = {int(n_layers * 0.6)}")

    captured = {}
    handles = []
    def make_hook(idx):
        def hook_fn(module, input, output):
            captured[idx] = output[0].detach()
        return hook_fn

    for i in range(n_layers):
        handles.append(model.model.layers[i].register_forward_hook(make_hook(i)))

    # 4. Verify chat template includes <think>
    print("\n[4/7] Verifying chat template...")
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    print(f"  Template ends with: ...'{prompt_text[-50:]}'")
    if "<think>" not in prompt_text:
        errors.append("Chat template does NOT include <think> — the self.thinking=True logic will be wrong")
    else:
        print("  CONFIRMED: template includes <think>, so thinking=True initialization is correct")

    # 5. Run a short generation and verify hook fires
    print("\n[5/7] Running short test generation (max 20 tokens)...")
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=20,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    if not captured:
        errors.append("No layer hooks fired during generation")
    else:
        print(f"  {len(captured)}/{n_layers} layer hooks fired")
        sample_shape = next(iter(captured.values())).shape
        last_dim = sample_shape[-1]
        print(f"  Sample hidden state shape: {sample_shape}")
        if last_dim != hidden_dim:
            errors.append(f"Hidden dim mismatch: config says {hidden_dim}, hook captured {last_dim}")
        else:
            print(f"  Hidden dim correct: {hidden_dim}")
        if len(captured) < n_layers:
            errors.append(f"Only {len(captured)}/{n_layers} hooks fired")

    for h in handles:
        h.remove()

    # 6. Verify logit access during generation
    print("\n[6/7] Verifying logit capture via hook on lm_head...")
    logit_captured = {}
    def logit_hook_fn(module, input, output):
        logit_captured["logits"] = output.detach()

    lm_handle = model.lm_head.register_forward_hook(logit_hook_fn)

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=5,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    lm_handle.remove()

    if "logits" not in logit_captured:
        errors.append("Logit hook on lm_head did NOT fire")
    else:
        lshape = logit_captured["logits"].shape
        print(f"  Logit hook fired. Shape: {lshape}")
        vocab_size = lshape[-1]
        print(f"  Vocab size from logits: {vocab_size}")

    # 7. Verify dataset loads
    print("\n[7/7] Verifying AIME dataset...")
    from datasets import load_dataset
    ds = load_dataset("gneubig/aime-1983-2024", split="train")
    print(f"  Total problems: {len(ds)}")
    print(f"  Columns: {ds.column_names}")
    ds_sample = ds.shuffle(seed=42).select(range(min(3, len(ds))))
    for r in ds_sample:
        print(f"  Sample Q (first 80 chars): {r['Question'][:80]}...")
        print(f"  Sample A: {r['Answer']}")
        break

    if "Question" not in ds.column_names or "Answer" not in ds.column_names:
        errors.append(f"Expected columns Question/Answer, got {ds.column_names}")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        print(f"  Model: {MODEL_NAME}")
        print(f"  Layers: {n_layers}, Hidden: {hidden_dim}")
        print(f"  Hook layer: {layer_idx}")
        print(f"  VRAM: {vram:.2f} GB")
        print(f"  Chat template includes <think>: YES")
        print(f"  Hook fires during generate: YES")
        print(f"  Logit hook fires: YES")
        print("\nSafe to proceed with: python generate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
