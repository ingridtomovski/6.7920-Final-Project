import os
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Configuration

PROJECT_DIR = Path.home() / "6.7920-Final-Project"
PROMPT_DIR = PROJECT_DIR / "data" / "prompts"
OUTPUT_FILE = PROJECT_DIR / "data" / "baseline_responses.json"

MODEL_NAME = "distilgpt2"
MAX_NEW_TOKENS = 150

# Load prompts

print("Loading prompts from:", PROMPT_DIR)

with open(PROMPT_DIR / "test_prompts.json", "r") as f:
    test_prompts = json.load(f)

print(f"Loaded {len(test_prompts)} test prompts")

# Load DistilGPT-2

print("Loading model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# DistilGPT-2 has no pad token; set to EOS to avoid warnings
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")
    device = "cuda"
else:
    device = "cpu"

print(f"Model loaded on {device}")

# Generate baseline responses

baseline_outputs = []

print("Generating baseline responses...")

for i, prompt in enumerate(test_prompts):
    print(f"Generating response {i+1}/{len(test_prompts)}...")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    baseline_outputs.append({
        "prompt": prompt,
        "response": response
    })

# ---------------------------------------
# Save to JSON
# ---------------------------------------

os.makedirs(PROJECT_DIR / "data" / "responses", exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(baseline_outputs, f, indent=2)

print("Saved baseline responses to:", OUTPUT_FILE)
print("Done!")
