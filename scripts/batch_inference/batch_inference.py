"""
batch_inference.py

Runs batch inference on a CSV of instruction-style prompts using the TinyLLaMA-1.1B model
with 8-bit quantization. The script processes input prompts in batches, generates responses
with controlled sampling parameters, and saves the output as a structured CSV.

Overview:

- Input: A CSV file with 'id' and 'prompt' columns.
- Output: A CSV file with 'id', 'prompt', and 'response' columns.
- Prompts are formatted in Alpaca-style ("### Instruction:" / "### Response:")
- Loops through prompts in batches with optional sleep delay between iterations.

Features:

- 8-bit quantized model loading (via BitsAndBytes) for memory efficiency
- Batched tokenization and inference for performance
- Sampling configuration: temperature, top-p, and max tokens
- Output saved in reproducible format for downstream training or analysis

Note:
This script is ideal for generating model responses at scale for prompt evaluation,
dataset creation, or synthetic data generation tasks.

"""


import pandas as pd
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

# ---------------------------
# Configuration
# ---------------------------

CSV_PATH = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/prompt_data/promptb3_rephrased.csv")
OUTPUT_PATH = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/prompt_data/all_results3_rephrased.csv")
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BATCH_SIZE = 10
SLEEP_SECONDS = 10
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9

# ---------------------------
# Load Tokenizer & Model (8-bit)
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
# Load Prompt Data
# ---------------------------

df = pd.read_csv(CSV_PATH)
total_prompts = len(df)
print(f"✅ Loaded {total_prompts} prompts from '{CSV_PATH}'\n")

# ---------------------------
# Process Prompts in Batches
# ---------------------------

all_results = []

for start in range(0, total_prompts, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total_prompts)
    batch = df.iloc[start:end]

    batch_num = (start // BATCH_SIZE) + 1

    print(f"🔹 Processing batch {batch_num} ({start + 1}-{end})")

    for _, row in batch.iterrows():
        prompt_id = row["id"]
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

        # Extract only the model response portion
        if "### Response:" in generated_text:
            final_response = generated_text.split("### Response:")[-1].strip()
        else:
            final_response = generated_text.strip()

        all_results.append({
            "id": int(prompt_id),
            "prompt": prompt_text,
            "response": final_response
        })

        # Print result
        print("-" * 40)
        print(f"Prompt:\n\n{prompt_text}\n")
        print(f"Answer:\n\n{final_response}\n")
        print("-" * 40)

    torch.cuda.empty_cache()

    if end < total_prompts:
        print(f"⏳ Sleeping {SLEEP_SECONDS} seconds before next batch...\n")
        time.sleep(SLEEP_SECONDS)

# ---------------------------
# Save All Results to CSV
# ---------------------------

output_df = pd.DataFrame(all_results)
output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"\n🎉 All prompts processed and saved to '{OUTPUT_PATH}'.")
