import torch
import json
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import PPOConfig, PPOTrainer
from datasets import Dataset

# =====================================================
# CONFIG
# =====================================================
policy_path  = "/Users/ingridtomovski/6.7920-Final-Project/models/llama3-1b"
reward_path  = "OpenAssistant/reward-model-deberta-v3-large-v2"
prompt_file  = "/Users/ingridtomovski/6.7920-Final-Project/data/prompts/dolly_prompts_train.json"
save_path    = "/Users/ingridtomovski/6.7920-Final-Project/models/ppo_llama1b"

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE  = torch.float32  # MPS does not fully support float16

# =====================================================
# LOAD DATA
# =====================================================
with open(prompt_file) as f:
    prompts = json.load(f)

dataset = Dataset.from_dict({"prompt": prompts})

# =====================================================
# TOKENIZER
# =====================================================
tokenizer = AutoTokenizer.from_pretrained(policy_path)
tokenizer.pad_token = tokenizer.eos_token

# =====================================================
# POLICY + REF + VALUE MODELS
# =====================================================
policy_model = AutoModelForCausalLM.from_pretrained(policy_path, torch_dtype=DTYPE).to(DEVICE)
ref_model    = AutoModelForCausalLM.from_pretrained(policy_path, torch_dtype=DTYPE).to(DEVICE)
ref_model.eval()
value_model  = copy.deepcopy(policy_model).to(DEVICE)

# =====================================================
# REWARD MODEL
# =====================================================
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_path,
    torch_dtype=DTYPE
).to(DEVICE)
reward_model.eval()

# =====================================================
# PPO CONFIG
# =====================================================
ppo_config = PPOConfig(
    batch_size=4,
    mini_batch_size=2,
    learning_rate=1e-6,
)

# =====================================================
# PPO TRAINER
# =====================================================
trainer = PPOTrainer(
    args=ppo_config,
    model=policy_model,
    ref_model=ref_model,
    value_model=value_model,
    train_dataset=dataset,
    processing_class=tokenizer,
    reward_model=reward_model
)

# =====================================================
# TRAIN LOOP (MPS-compatible)
# =====================================================
print("\nRunning PPO...")

for step, prompt_text in enumerate(dataset["prompt"][:20]):  # small test run
    # 1️. Tokenize prompt and move to DEVICE
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 2️. Generate response with raw policy model
    with torch.inference_mode():
        response_ids = policy_model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
        )
    # Remove prompt tokens from the output
    response_text = tokenizer.decode(response_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # 3️. Compute reward
    reward_inputs = tokenizer(response_text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.inference_mode():
        reward_score = reward_model(**reward_inputs).logits.mean()
    reward_tensor = torch.tensor([reward_score], dtype=torch.float32, device=DEVICE)

    # 4️. PPO update
    trainer.step([prompt_text], [response_text], reward_tensor)

    print(f"\nStep {step+1}: reward = {reward_score:.4f}")
    print(response_text)

# =====================================================
# SAVE MODEL
# =====================================================
trainer.save_model(save_path)
print("\nPPO Training Finished — Model Saved!")
