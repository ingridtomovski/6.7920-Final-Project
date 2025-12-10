import os
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ================================
# CONFIG
# ================================
PROJECT_DIR = Path.home() / "6.7920-Final-Project"
PROMPTS_FILE = PROJECT_DIR / "data/prompts/test_prompts.json"
OUTPUT_FILE = PROJECT_DIR / "data/responses/tinyllama_responses.json"

# Change this to a TinyLlama checkpoint; e.g. the chat‑tuned one on HF
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 50
BATCH_SIZE = 8   # batch size for prompt generation if you want chunked batching

# ================================
print("Loading prompts:", PROMPTS_FILE)
with open(PROMPTS_FILE, "r") as f:
    prompts = json.load(f)
print(f"Loaded {len(prompts)} prompts.\n")

# -------------------------------
# Load Model + Tokenizer
# -------------------------------
print("Loading TinyLlama model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,      # or torch.float32 if float16 fails
    device_map="auto"               # adapt to available device (GPU/CPU)
)

# Use a HuggingFace pipeline for convenience
text_gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    dtype=torch.float16,
    # you can set default kwargs here if you like
)

# -------------------------------
# Generate responses
# -------------------------------
generated = []
print("Generating responses with TinyLlama...\n")

for i, prompt in enumerate(prompts):
    print(f" → {i+1}/{len(prompts)}", end="\r")

    # Format prompt if using chat style (optional)
    # For a plain prompt, you could just use prompt
    formatted = prompt
    # If you want to encourage chat‑style responses, you could wrap:
    # formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    out = text_gen(
        formatted,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        return_full_text=False,        # i.e. return only the generated continuation
        # repetition_penalty=1.1,      # optionally
    )

    resp = out[0]["generated_text"]
    generated.append({"prompt": prompt, "response": resp})

# -------------------------------
# Save Output
# -------------------------------
os.makedirs(PROJECT_DIR / "data/responses", exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(generated, f, indent=2)

print("\n\nTinyLlama generation complete!")
print("Responses saved to:", OUTPUT_FILE)
