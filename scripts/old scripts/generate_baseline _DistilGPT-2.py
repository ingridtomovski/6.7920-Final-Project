import os
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================================
# CONFIG
# ================================
PROJECT_DIR = Path.home() / "6.7920-Final-Project"
PROMPTS_FILE = PROJECT_DIR / "data/prompts/test_prompts.json"
OUTPUT_FILE = PROJECT_DIR / "data/responses/baseline_responses.json"

MODEL_NAME = "distilgpt2"
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7
BATCH_SIZE = 8   # Set to 1 if you want slower/safer generation
# ================================

print("Loading prompts:", PROMPTS_FILE)
with open(PROMPTS_FILE, "r") as f:
    prompts = json.load(f)

print(f"Loaded {len(prompts)} test prompts.\n")

# -------------------------------
# Load Model + Tokenizer
# -------------------------------
print("Loading baseline model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# # distilgpt2 doesn't define pad_token → use EOS instead
# tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Resize model embeddings to account for the new pad token
model.resize_token_embeddings(len(tokenizer))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

print(f"Model loaded on {device}\n")

# -------------------------------
# Generate Responses
# -------------------------------
generated = []
print("Generating baseline responses...\n")

for i, prompt in enumerate(prompts):
    print(f" → {i+1}/{len(prompts)}", end="\r")

    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    outputs = model.generate(
        inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        pad_token_id=tokenizer.pad_token_id
    )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    generated.append({"prompt": prompt, "response": response_text})


# -------------------------------
# Save Output
# -------------------------------
os.makedirs(PROJECT_DIR / "data/responses", exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(generated, f, indent=2)

print("\n\n Baseline generation complete!")
print("Responses saved to:", OUTPUT_FILE)
