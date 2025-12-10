import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax
import os
from pathlib import Path

# Configuration

PROJECT_DIR = Path.home() / "6.7920-Final-Project"
BASELINE_FILE = PROJECT_DIR / "data" / "responses" / "baseline_responses.json"   # <-- your generated responses
OUTPUT_FILE = PROJECT_DIR / "data/rewards/baseline_rewards.json"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Load sentiment model

print(f"Loading reward model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Load baseline responses

with open(BASELINE_FILE, "r") as f:
    responses = json.load(f)

print(f"Loaded {len(responses)} baseline responses.")

# Reward function

def compute_reward(text):
    """
    Returns a scalar reward using sentiment probabilities.
    reward = positive_prob - negative_prob
    """

    # Encode input
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**encoded)

    logits = output.logits[0].numpy()
    probabilities = softmax(logits)

    neg, neu, pos = probabilities

    reward = pos - neg           # a simple, effective signal

    return {
        "negative": float(neg),
        "neutral": float(neu),
        "positive": float(pos),
        "reward": float(reward)
    }

# Score all responses

results = []

print("Scoring responses...")
for item in responses:
    prompt = item["prompt"]
    response = item["response"]
    scores = compute_reward(response)

    results.append({
        "prompt": prompt,
        "response": response,
        "scores": scores
    })

# Save reward results

os.makedirs(PROJECT_DIR / "data/rewards", exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved reward scores to {OUTPUT_FILE}!")
