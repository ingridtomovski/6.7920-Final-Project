from datasets import load_dataset
import json
import random

# --------- CONFIG ---------
NUM_PROMPTS = 400        # Change this
TRAIN_RATIO = 0.75       # 75% train / 25% test
OUTPUT_FILE = "/Users/ingridtomovski/6.7920-Final-Project/data/prompts/dolly_prompts"  # File name prefix
# --------------------------

print("Loading dataset...")
ds = load_dataset("databricks/databricks-dolly-15k")["train"]

# Extract instruction field as prompt
prompts = [item["instruction"].strip() for item in ds if item["instruction"]]

# Remove duplicates
prompts = list(dict.fromkeys(prompts))  # preserves order

# Trim to desired size
prompts = prompts[:NUM_PROMPTS]

# Shuffle before split
random.shuffle(prompts)

train_size = int(len(prompts) * TRAIN_RATIO)
train_prompts = prompts[:train_size]
test_prompts = prompts[train_size:]

# Save to json
with open(f"{OUTPUT_FILE}_train.json", "w") as f:
    json.dump(train_prompts, f, indent=2)

with open(f"{OUTPUT_FILE}_test.json", "w") as f:
    json.dump(test_prompts, f, indent=2)

print(f"Saved {len(train_prompts)} training prompts → {OUTPUT_FILE}_train.json")
print(f"Saved {len(test_prompts)} testing prompts → {OUTPUT_FILE}_test.json")
