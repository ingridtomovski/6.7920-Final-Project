import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# --- CONFIG ---
input_json_file = "/Users/ingridtomovski/6.7920-Final-Project/data/prompts/dolly_prompts_test.json"           # Input JSON with prompts
output_json_file = "/Users/ingridtomovski/6.7920-Final-Project/data/responses/baseline_responses_dolly.json"        # Output JSON file
model_path = "/Users/ingridtomovski/6.7920-Final-Project/models/llama3-1b" # Path to your local LLaMA-3.2-1B folder
max_length = 200
temperature = 0.7
top_p = 0.9

# --- LOAD MODEL ---
print("Loading tokenizer and model from local folder...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True  # needed for LLaMA-safetensors
)
model.eval()
print("Model loaded successfully.")

# --- READ INPUT JSON ---
with open(input_json_file, "r") as f:
    prompts = json.load(f)

# --- GENERATE RESPONSES ---
output_data = []
for prompt in tqdm(prompts, desc="Generating responses"):
    full_prompt = (
        f"{prompt}\n\n"
        "Instructions for the AI: Answer in 3-5 complete sentences. "
        "Be clear, relevant, and avoid repeating the prompt. "
        "Provide examples or explanations when appropriate."
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.2
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- Strip full_prompt from the start ---
    if decoded.startswith(full_prompt):
        response_text = decoded[len(full_prompt):].strip()
    else:
        response_text = decoded

    output_data.append({
        "prompt": prompt,
        "response": response_text
    })

# --- SAVE OUTPUT JSON ---
with open(output_json_file, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"All responses saved to {output_json_file}")
