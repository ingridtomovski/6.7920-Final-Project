from datasets import load_dataset
import json
import random

# =============== SETTINGS =============== #
TRAIN_SIZE = 300
TEST_SIZE  = 100
SAVE_DIR   = "/Users/ingridtomovski/6.7920-Final-Project/data/prompts"
# ======================================== #

# 1. Load dataset
ds = load_dataset("nvidia/HelpSteer2", split="train")

# 2. Collect unique text prompts
unique_prompts = list({item["prompt"] for item in ds})  # set() deduplicates

print(f"Total unique prompts found: {len(unique_prompts)}")

# Check dataset is large enough
assert len(unique_prompts) >= TRAIN_SIZE + TEST_SIZE, "Not enough unique prompts!"

# 3. Shuffle for random selection
random.shuffle(unique_prompts)

# 4. Split train + test with no overlap
train_prompts = unique_prompts[:TRAIN_SIZE]
test_prompts  = unique_prompts[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]

# 5. Save to JSON files
import os
os.makedirs(SAVE_DIR, exist_ok=True)

with open(f"{SAVE_DIR}/train_prompts.json", "w") as f:
    json.dump(train_prompts, f, indent=2)

with open(f"{SAVE_DIR}/test_prompts.json", "w") as f:
    json.dump(test_prompts, f, indent=2)

print("Saved train_prompts.json and test_prompts.json successfully!")
