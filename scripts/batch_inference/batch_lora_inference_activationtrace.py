"""
batch_lora_inference_activationtrace.py

Performs batched inference using a TinyLLaMA language model with an injected LoRA adapter
for instruction-response generation. Captures mean layer-wise activations for each prompt
and saves them as structured JSON files for further analysis.

Key Features:
- Uses BitsAndBytes 8-bit quantization for VRAM efficiency.
- Injects custom LoRA adapter (fine-tuned on Python instruction pairs).
- Captures activations for each transformer layer post-LoRA.
- Outputs responses and activation logs in sync per prompt.
- Includes configurable throttling between prompts and batches to manage GPU load.

"""

import pandas as pd
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
import json
from peft import PeftModel

# ---------------------------
# Configuration
# ---------------------------

CSV_PATH = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/prompt_data/promptb2.csv")
OUTPUT_PATH = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/prompt_data/results/all_results2_lora_activation.csv")
ACTIVATION_DIR = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/logs/activation_logs/post_LoRa/raw/pythonbasic_100")
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_ADAPTER_PATH = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/lora_adapters/python_light")
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
# Load Model + LoRA
# ---------------------------

print("🧠 Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="cuda",
    torch_dtype=torch.float16
)

print("🔌 Plugging in LoRA adapters...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
print("✅ LoRA activated.\n")

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

for i, block in enumerate(model.base_model.model.model.layers):
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

        # Extract only the response part
        if "### Response:" in generated_text:
            final_response = generated_text.split("### Response:")[-1].strip()
        else:
            final_response = generated_text.strip()

        all_results.append({
            "id": prompt_id,
            "prompt": prompt_text,
            "response": final_response
        })

        # Serialize and save activations
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

        # Clear activations and cache
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
