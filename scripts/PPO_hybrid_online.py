import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, GenerationConfig
from trl import PPOConfig, AutoModelForCausalLMWithValueHead, PPOTrainer
from tqdm import tqdm
import copy


# ================================
# CONFIG
# ================================
POLICY_PATH   = "/Users/ingridtomovski/6.7920-Final-Project/models/llama3-1b"
REWARD_MODEL  = "OpenAssistant/reward-model-deberta-v3-large-v2"

ROUNDS        = 5
GENERATE_N    = 80
OUTPUT_DIR    = "/Users/ingridtomovski/6.7920-Final-Project/models/ppo_rounds/"
PROMPT_FILE   = "/Users/ingridtomovski/6.7920-Final-Project/data/prompts/dolly_prompts_train.json"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# ===========================
# LOAD POLICY (VALUE HEAD)
# ===========================
tokenizer = AutoTokenizer.from_pretrained(POLICY_PATH)
tokenizer.pad_token = tokenizer.eos_token

policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    POLICY_PATH, torch_dtype=torch.float16 if DEVICE!="cpu" else torch.float32
).to(DEVICE)

policy.generation_config = GenerationConfig.from_model_config(policy.config)

if not hasattr(policy, "base_model_prefix"):
    policy.base_model_prefix = "model"

# Reference model for KL penalty
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    POLICY_PATH, torch_dtype=torch.float16 if DEVICE!="cpu" else torch.float32
).to(DEVICE)
ref_model.eval()

# ===========================
# LOAD REWARD MODEL
# ===========================
reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL, torch_dtype=torch.float16 if DEVICE!="cpu" else torch.float32
).to(DEVICE)
reward_model.eval()


# ===========================
# PPO CONFIG
# ===========================
ppo_config = PPOConfig(
    learning_rate=1e-6,
    batch_size=32,
    mini_batch_size=16,
)


# ===========================
# Generate response
# ===========================
def generate_response(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = policy.generate(
            **input_ids,
            max_new_tokens=200,
            temperature=0.8,
            top_p=0.9,
            do_sample=True
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# ===========================
# Reward scoring
# ===========================
def compute_reward(prompt, response):
    text = f"Prompt: {prompt}\nResponse: {response}"
    tokens = reward_tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        return reward_model(**tokens).logits.cpu().item()


# ===========================
# ONLINE PPO LOOP
# ===========================
with open(PROMPT_FILE) as f:
    prompts = json.load(f)

for round in range(1, ROUNDS+1):
    print(f"\n========== ROUND {round}/{ROUNDS} ==========")

    batch_prompts, batch_responses, batch_rewards = [], [], []

    # ---- Generate & score new batch ----
    for prompt in tqdm(prompts[:GENERATE_N]):
        resp = generate_response(prompt)
        reward = compute_reward(prompt, resp)
        batch_prompts.append(prompt)
        batch_responses.append(resp)
        batch_rewards.append(reward)

    ds = Dataset.from_dict({
        "query": batch_prompts,
        "response": batch_responses,
        "rewards": batch_rewards
    })

    # ---- Train PPO on this round ----
    trainer = PPOTrainer(
        args=ppo_config,
        model=policy,
        ref_model=ref_model,
        processing_class=tokenizer,
        train_dataset=ds,
        reward_model=reward_model,
        value_model=policy.v_head
    )

    trainer.train()

    # ---- Save checkpoint ----
    save_dir = os.path.join(OUTPUT_DIR, f"round_{round}")
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved PPO checkpoint → {save_dir}")
