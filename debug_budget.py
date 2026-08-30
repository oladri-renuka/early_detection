"""Debug: print raw generation at budget=256 for first problem."""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
import random

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
BUDGET = 256

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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, device_map="auto")
model.eval()

think_end_id = tokenizer.convert_tokens_to_ids("</think>")
ds = list(load_dataset("gneubig/aime-1983-2024", split="train"))
rng = random.Random(0)
problems = rng.sample(ds, 200)
item = problems[0]

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": item["Question"]}],
    tokenize=False, add_generation_prompt=True
) + "<think>\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
processor = BudgetForcingProcessor(think_end_id, BUDGET)

with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=BUDGET+200, do_sample=False,
        logits_processor=[processor]
    )

generated = out[0][inputs["input_ids"].shape[1]:]
text = tokenizer.decode(generated, skip_special_tokens=False)
print("=== GENERATED TEXT ===")
print(repr(text[-500:]))  # last 500 chars
print("=== END ===")
print("Contains </think>:", "</think>" in text)
print("Contains \\boxed:", "\\boxed" in text)
