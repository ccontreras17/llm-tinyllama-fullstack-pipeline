"""
batch_inference_activationtrace.py

Runs batch inference on instruction-style prompts using the TinyLLaMA-1.1B model with 8-bit quantization,
while capturing and saving intermediate activation values for each transformer layer.

Overview:

- Loads prompts from a CSV file and formats them in Alpaca-style ("### Instruction:" / "### Response:")
- Registers forward hooks to record activation outputs from each transformer block during generation
- For each prompt, performs inference and extracts the generated response
- Saves mean activation values (per layer) as JSON files for downstream analysis
- Outputs all responses in a final CSV file

Use Case:

This script is intended for model interpretability analysis, activation tracing, and layer-wise representation studies.
It supports batch processing with sleep intervals to reduce GPU strain and maintain stable throughput.

Model: TinyLLaMA-1.1B-Chat-v1.0 (8-bit via BitsAndBytes)
Output: Response CSV + Per-prompt activation logs (JSON)
Activation Format: Mean of each layer’s output tensor, averaged across (batch, sequence)

"""

import pandas as pd
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
import json

# ---------------------------
# Configuration
# ---------------------------

CSV_PATH = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/prompt_data/promptb3.csv")
OUTPUT_PATH = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/prompt_data/results/all_results3_activation.csv")
ACTIVATION_DIR = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/logs/activation_logs/pre_LoRa/raw/pythonbasicplus_100")
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BATCH_SIZE = 10
SLEEP_SECONDS = 10
SLEEP_BETWEEN_PROMPTS = 2
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9

# ---------------------------
# Create Directory for Activations
# ---------------------------

os.makedirs(ACTIVATION_DIR, exist_ok=True)

# ---------------------------
# Load Tokenizer and Model
# ---------------------------

print("🧠 Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="cuda",
    torch_dtype=torch.float16
)

print("✅ Model loaded.\n")

# ---------------------------
# Register Forward Hooks to Capture Activations
# ---------------------------

activations = {}

def save_activation(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            tensor = output[0]
        else:
            tensor = output
        activations[name] = tensor.detach().cpu()
    return hook

for i, block in enumerate(model.model.layers):
    block.register_forward_hook(save_activation(f"layer_{i}"))

# ---------------------------
# Load Prompt Dataset
# ---------------------------

df = pd.read_csv(CSV_PATH)
total_prompts = len(df)
print(f"✅ Loaded {total_prompts} prompts from '{CSV_PATH}'\n")

# ---------------------------
# Run Inference and Capture Activations
# ---------------------------

all_results = []

for start in range(0, total_prompts, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total_prompts)
    batch = df.iloc[start:end]

    batch_num = (start // BATCH_SIZE) + 1

    print(f"🔹 Processing batch {batch_num} ({start + 1}-{end})")

    for _, row in batch.iterrows():
        prompt_id = int(row["id"])
        prompt_text = row["prompt"]

        formatted_prompt = f"### Instruction:\n{prompt_text}\n\n### Response:\n"

        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response segment
        if "### Response:" in generated_text:
            final_response = generated_text.split("### Response:")[-1].strip()
        else:
            final_response = generated_text.strip()

        all_results.append({
            "id": prompt_id,
            "prompt": prompt_text,
            "response": final_response
        })

        # Serialize and save mean activations per layer
        activations_serializable = {
            k: v.mean(dim=(0, 1)).tolist()
            for k, v in activations.items()
        }

        activation_path = os.path.join(ACTIVATION_DIR, f"activation_{prompt_id:05}.json")
        with open(activation_path, "w") as f:
            json.dump({
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "activations": activations_serializable
            }, f)

        # Clear stored activations and free memory
        activations.clear()
        torch.cuda.empty_cache()

        # Determine whether to sleep between prompts
        is_last_prompt_in_batch = (row.name == batch.index[-1])
        is_first_prompt_overall = (start == 0 and row.name == batch.index[0])

        if not is_last_prompt_in_batch and not is_first_prompt_overall:
            print(f"⏳ Sleeping {SLEEP_BETWEEN_PROMPTS}s before next prompt...")
            time.sleep(SLEEP_BETWEEN_PROMPTS)

        # Output preview
        print("-" * 40)
        print(f"Prompt:\n\n{prompt_text}\n")
        print(f"Answer:\n\n{final_response}\n")
        print("-" * 40)

    torch.cuda.empty_cache()

    if end < total_prompts:
        print(f"⏳ Sleeping {SLEEP_SECONDS}s before next batch...\n")
        time.sleep(SLEEP_SECONDS)

# ---------------------------
# Save All Prompt-Response Pairs to CSV
# ---------------------------

output_df = pd.DataFrame(all_results)
output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"\n🎉 All prompts processed and saved to '{OUTPUT_PATH}'.")
