import json
from datasets import load_dataset
import os

# Configuration

N_TRAIN = 500      # number of examples to use for training PPO
N_TEST = 100       # held-out examples for evaluation
DATASET_NAME = "Anthropic/hh-rlhf"
SUBSET = "harmless-base"   # options: harmless-base, red-team-attempts, etc.
OUTPUT_DIR = "/Users/ingridtomovski/6.7920-Final-Project/data/prompts"

# Load Dataset

print(f"Loading dataset: {DATASET_NAME} ({SUBSET})...")
dataset = load_dataset(DATASET_NAME, data_dir=SUBSET)

train_split = dataset["train"]
print(f"Dataset loaded! Total training samples: {len(train_split)}")

def extract_prompt(text):
    """
    Given a 'chosen' or 'rejected' text from HH-RLHF, extract the user prompt.
    Assumes the format:
        Human: <prompt>
        Assistant: <response>
    """
    lines = text.strip().split("\n")
    for line in lines:
        if line.startswith("Human:"):
            # Remove "Human: " prefix and return
            return line[len("Human: "):].strip()
    return None

# Extract Prompts Only

print("Extracting prompts...")
prompts = [extract_prompt(example["chosen"]) for example in train_split]

print(f"Total prompts available: {len(prompts)}")

# Subsample for local training

train_data = prompts[:N_TRAIN]
test_data = prompts[N_TRAIN:N_TRAIN+N_TEST]

print(f"Using {len(train_data)} training prompts and {len(test_data)} test prompts.")

# Save to disk

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "train_prompts.json"), "w") as f:
    json.dump(train_data, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "test_prompts.json"), "w") as f:
    json.dump(test_data, f, indent=2)

print("Saved:")
print("- data/train_prompts.json")
print("- data/test_prompts.json")
print("Done!")
