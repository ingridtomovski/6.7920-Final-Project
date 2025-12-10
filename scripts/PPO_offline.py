# ppo_test_fixed_generation_config.py
import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, GenerationConfig, AutoModelForSequenceClassification, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import types
# ================================
# PATHS / DEVICE
# ================================
POLICY_PATH = "/Users/ingridtomovski/6.7920-Final-Project/models/llama3-1b"
REWARD_MODEL = "OpenAssistant/reward-model-deberta-v3-large-v2"
PRECOMPUTED_FILE = "/Users/ingridtomovski/6.7920-Final-Project/data/rewards/reward_scores_dolly.json"
OUTPUT_DIR = "/Users/ingridtomovski/6.7920-Final-Project/models/ppo_test_small/"
MAX_LEN = 512
BATCH_SIZE = 4

DEVICE = ("cuda" if torch.cuda.is_available() else
          "mps" if torch.backends.mps.is_available() else "cpu")

# ================================
# LOAD MODELS + TOKENIZER
# ================================
tokenizer = AutoTokenizer.from_pretrained(POLICY_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# policy, ref, and value should be full models (not submodules)
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    POLICY_PATH, torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32
).to(DEVICE)

value_model = policy_model  # can reuse policy for offline PPO

# Separate ref model (no value head!)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    POLICY_PATH, torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32
).to(DEVICE)
ref_model.eval()

# Some TRL/transformers combos expect generation_config & base_model_prefix.
# Add them if missing.
for m in (policy_model, ref_model, value_model):
    if not hasattr(m, "generation_config"):
        # Create generation_config from the model config
        m.generation_config = GenerationConfig.from_model_config(m.config)
    if not hasattr(m, "base_model_prefix"):
        m.base_model_prefix = "model"

# Reward model
reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL,
    torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32
).to(DEVICE)
reward_model.eval()

# ================================
# LOAD PRECOMPUTED DATA (tiny subset)
# ================================

with open(PRECOMPUTED_FILE, "r") as f:
    data = json.load(f)

data = data[:10]  # small sample
print(f"Loaded {len(data)} examples for dry-run")

prompts = [d["prompt"] for d in data]      # list of prompt strings
responses = [d["response"] for d in data]  # list of response strings
rewards = [float(d["score"]) for d in data]  # list of floats

# Combine prompt + response for PPO input
texts = [p + tokenizer.eos_token + r for p, r in zip(prompts, responses)]

encodings = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
    return_tensors=None
)

# Build HuggingFace dataset
ppo_dataset = Dataset.from_dict({
    "input_ids": encodings["input_ids"],
    "attention_mask": encodings["attention_mask"],
    "reward": rewards
})

# ================================
# PPO CONFIG (tiny)
# ================================
ppo_config = PPOConfig(
    batch_size=5,
    mini_batch_size=2,
    learning_rate=5e-6
)

# ================================
# CREATE TRAINER (pass reward_model & value_model)
# ================================
def ensure_backbone_alias(m):
    if hasattr(m, "pretrained_model") and not hasattr(m, "model"):
        print("⚠ Fixing model: creating m.model alias → m.pretrained_model")
        m.model = m.pretrained_model

for m in (policy_model, ref_model, value_model):
    ensure_backbone_alias(m)

for m in (policy_model, ref_model, value_model):
    if not hasattr(m, "is_gradient_checkpointing"):
        print("⚠ Fix: adding m.is_gradient_checkpointing=False")
        m.is_gradient_checkpointing = False

trainer = PPOTrainer(
    args=ppo_config,
    model=policy_model,
    ref_model=ref_model,
    processing_class=tokenizer,
    train_dataset=ppo_dataset,
    reward_model=reward_model,
    value_model=value_model,
)

print("\nStarting micro PPO training...\n")
trainer.train()
print("\nPPO DRY RUN COMPLETE\n")

# ================================
# SAVE
# ================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
policy_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved → {OUTPUT_DIR}")
