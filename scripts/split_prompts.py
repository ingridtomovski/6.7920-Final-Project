import json
import random
from pathlib import Path

# === CONFIG ===
PROJECT_DIR = Path.home() / "6.7920-Final-Project"
PROMPT_FILE = PROJECT_DIR / "data/prompts/self_generated_prompts.json"   # your dataset

TRAIN_OUT = PROJECT_DIR / "data/prompts/train_prompts.json"
TEST_OUT  = PROJECT_DIR / "data/prompts/test_prompts.json"

TRAIN_SIZE = 300
TEST_SIZE  = 100

# === LOAD ===
print(f"Loading prompts file: {PROMPT_FILE}")

with open(PROMPT_FILE, "r") as f:
    data = json.load(f)

# your data structure = { "ppo_prompts": [{id, prompt}, ...]}
prompts = [item["prompt"] for item in data]

print(f"Extracted {len(prompts)} total prompts.")

# === CHECK ENOUGH DATA ===
assert len(prompts) >= TRAIN_SIZE + TEST_SIZE, \
    f"Not enough prompts — need {TRAIN_SIZE + TEST_SIZE}, only have {len(prompts)}"

# === SHUFFLE & SPLIT ===
random.shuffle(prompts)

train_prompts = prompts[:TRAIN_SIZE]
test_prompts  = prompts[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]

# === SAVE ===
TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)

with open(TRAIN_OUT, "w") as f:
    json.dump(train_prompts, f, indent=2)

with open(TEST_OUT, "w") as f:
    json.dump(test_prompts, f, indent=2)

print(f"\n✓ Saved {TRAIN_SIZE} training prompts  → {TRAIN_OUT}")
print(f"✓ Saved {TEST_SIZE} test prompts      → {TEST_OUT}\n")
print("Done!")
