"""
batch_inference_rag_activationtrace.py

Performs retrieval-augmented generation (RAG) with a quantized TinyLLaMA-1.1B model and logs
intermediate layer activations during inference for each prompt.

Overview:
- Loads a CSV of instruction-style prompts.
- Retrieves semantically similar examples using a SentenceTransformer embedding model
  (all-MiniLM-L6-v2) running on CPU.
- Constructs an in-context prompt using the top-k (k=1) retrieved example(s) for each input.
- Generates responses using TinyLLaMA-1.1B-Chat-v1.0, quantized to 8-bit via BitsAndBytes, running on GPU.
- Captures mean activation values per transformer layer via registered forward hooks.
- Saves both the model responses and activation traces for every prompt.

Use Case:
This script is built for interpretability research and embedding-aware inference,
allowing analysis of layer-wise behavior under RAG-enhanced prompting strategies.

Outputs:
- `all_results3_rag_activation.csv` — Prompt-response pairs
- Individual JSON files — Layer-averaged activation logs per prompt

Note:
Only the top-1 retrieved example is included in the context to minimize attention bottlenecks
and reduce token truncation on memory-constrained models like TinyLLaMA.

"""

import pandas as pd
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
import torch
import json
import os

# ---------------------------
# Configuration
# ---------------------------

CSV_PATH = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/prompt_data/promptb3.csv")
OUTPUT_PATH = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/prompt_data/all_results3_rag_activation.csv")
ACTIVATION_DIR = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/logs/activation_logs/pre_LoRa/rag/pythonbasicplus_100")
RAG_JSON = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/rag/200_python_basics.json")
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BATCH_SIZE = 10
SLEEP_PER_PROMPT = 5
SLEEP_PER_BATCH = 20
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K_RETRIEVAL = 1

# ---------------------------
# Setup
# ---------------------------

os.makedirs(ACTIVATION_DIR, exist_ok=True)

print("📖 Loading retrieval examples...")
with open(RAG_JSON, "r") as f:
    examples = json.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
example_texts = [ex["question"] for ex in examples]
example_embeddings = embedder.encode(
    example_texts,
    convert_to_tensor=True,
    batch_size=32,
    show_progress_bar=True
)
print(f"✅ Loaded and embedded {len(examples)} retrieval examples.\n")

print("🧠 Loading TinyLlama model...")
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
        activations[name] = output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu()
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

    for i, row in batch.iterrows():
        prompt_id = int(row["id"])
        prompt_text = row["prompt"]

        # Retrival
        prompt_embedding = embedder.encode(prompt_text, convert_to_tensor=True)
        similarities = util.cos_sim(prompt_embedding, example_embeddings)[0]
        top_indices = torch.topk(similarities, k=TOP_K_RETRIEVAL).indices

        retrieved_chunks = [
            f"###Example\n### Instruction:\n{examples[idx]['question']}\n\n### Response:\n{examples[idx]['answer']}"
            for idx in top_indices
        ]
        retrieval_context = "\n\n".join(retrieved_chunks)

        # Format
        final_prompt = (
            f"{retrieval_context}\n\n"
            f"### Instruction:\n{prompt_text.strip()}\n\n"
            f"### Response:\n"
        )


        inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Response:" in decoded:
            final_response = decoded.split("### Response:")[-1].split("### Instruction:")[0].strip()
        else:
            final_response = decoded.strip()

        # Save Activation
        activations_serializable = {
            k: v.mean(dim=(0, 1)).tolist()
            for k, v in activations.items()
        }

        with open(os.path.join(ACTIVATION_DIR, f"activation_{prompt_id:05}.json"), "w") as f:
            json.dump({
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "activations": activations_serializable
            }, f)

        # Add to results
        all_results.append({
            "id": prompt_id,
            "prompt": prompt_text,
            "response": final_response
        })


        activations.clear()
        torch.cuda.empty_cache()

        # Log
        print("-" * 40)
        print(f"Prompt:\n{prompt_text}\n")
        print(f"Answer:\n{final_response}\n")
        print("-" * 40)

        if i != batch.index[0] and i != batch.index[-1]:
            print(f"⏳ Sleeping {SLEEP_PER_PROMPT} seconds before next prompt...\n")
            time.sleep(SLEEP_PER_PROMPT)

    if end < total_prompts:
        print(f"⏳ Batch complete. Sleeping {SLEEP_PER_BATCH} seconds before next batch...\n")
        time.sleep(SLEEP_PER_BATCH)

# ---------------------------
# Save All Prompt-Response Pairs to CSV
# ---------------------------

pd.DataFrame(all_results).to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"\n🎉 All prompts processed and saved to '{OUTPUT_PATH}'.")
