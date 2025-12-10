import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========================================
# CONFIG
# ========================================
INPUT_FILE = "/Users/ingridtomovski/6.7920-Final-Project/data/responses/baseline_responses_dolly.json"     # your saved generations
OUTPUT_FILE = "/Users/ingridtomovski/6.7920-Final-Project/data/rewards/reward_scores_dolly.json"
MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"
BATCH_SIZE = 8

# ========================================
# Load Reward Model
# ========================================
print(f"Loading reward model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.eval()

print("Model loaded!\n")

# ========================================
# Load Generated Responses JSON
# Must have fields: {"prompt": "...", "response": "..."}
# ========================================
print("Reading input response file:", INPUT_FILE)
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

prompts = [d["prompt"] for d in data]
responses = [d["response"] for d in data]

print(f"Loaded {len(prompts)} responses.\n")

# ========================================
# Scoring Function
# ========================================
def reward_score_batch(prompts, responses, batch_size=8):
    scores = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_responses = responses[i:i+batch_size]

        batch_texts = [
            f"Prompt: {p}\nResponse: {r}"
            for p, r in zip(batch_prompts, batch_responses)
        ]

        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            logits = model(**tokens).logits.squeeze(-1)

        scores.extend(logits.cpu().tolist())  # convert to python floats

    return scores


# ========================================
# Compute Scores
# ========================================
print("Scoring responses...")
scores = reward_score_batch(prompts, responses, BATCH_SIZE)
print("Done!\n")

# ========================================
# Save as JSON with prompt+response+score
# ========================================
scored_output = [
    {"prompt": p, "response": r, "score": s}
    for p, r, s in zip(prompts, responses, scores)
]

with open(OUTPUT_FILE, "w") as f:
    json.dump(scored_output, f, indent=2)

print(f"Scores saved → {OUTPUT_FILE}")
