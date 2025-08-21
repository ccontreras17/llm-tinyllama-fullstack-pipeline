"""
batch_inference_rag.py

Performs batch Retrieval-Augmented Generation (RAG) using a TinyLLaMA-1.1B model with 8-bit quantization.
Each prompt in the input CSV is augmented with semantically similar examples retrieved from a reference dataset,
formatted into a few-shot prompt, and passed to the model for response generation.

Overview:

- Loads a reference set of question-answer pairs (`.json`) and encodes them using SentenceTransformers.
- For each input prompt, retrieves the top-k similar examples (k=1) based on cosine similarity.
- Constructs a prompt using retrieved examples and formats it in a question-answer style.
- Runs inference in batches with optional sleep intervals to manage GPU load.
- Saves all model outputs as a structured `.csv` file with original prompt and generated response.

Retrieval Model: `all-MiniLM-L6-v2` (runs on CPU)
Language Model: `TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0` (8-bit, runs on CUDA)
Top-k Retrieval: 1 (to respect the model’s limited context window and optimize relevance)

This script is ideal for large-scale synthetic dataset generation, fine-tuning pipelines, or prompt testing
in resource-constrained environments.

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

CSV_PATH = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/prompt_data/promptb3_rephrased.csv")
OUTPUT_PATH = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/prompt_data/all_results3_rephrased_rag.csv")
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
# Load Retrieval Examples
# ---------------------------

print("📖 Loading retrieval examples...")
with open(RAG_JSON, "r") as f:
    examples = json.load(f)

# Encode retrieval examples using CPU-based semantic model
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
example_texts = [ex["question"] for ex in examples]
example_embeddings = embedder.encode(example_texts, convert_to_tensor=True)

print(f"✅ Loaded {len(examples)} retrieval examples.\n")

# ---------------------------
# Load Model
# ---------------------------

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
# Load Prompts to Process
# ---------------------------

df = pd.read_csv(CSV_PATH)
total_prompts = len(df)
print(f"✅ Loaded {total_prompts} prompts from '{CSV_PATH}'\n")

# ---------------------------
# Batch Inference with RAG
# ---------------------------

all_results = []

for start in range(0, total_prompts, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total_prompts)
    batch = df.iloc[start:end]
    batch_num = (start // BATCH_SIZE) + 1

    print(f"🔹 Processing batch {batch_num} ({start + 1}-{end})")

    for i, row in batch.iterrows():
        prompt_id = row["id"]
        prompt_text = row["prompt"]

        # Encode current prompt and find similar example(s)
        prompt_embedding = embedder.encode(prompt_text, convert_to_tensor=True)
        similarities = util.cos_sim(prompt_embedding, example_embeddings)[0]
        top_indices = torch.topk(similarities, k=TOP_K_RETRIEVAL).indices

        retrieved_chunks = []
        for idx in top_indices:
            ex = examples[idx]
            retrieved_chunks.append(f"Q: {ex['question']}\nA: {ex['answer']}")

        retrieval_context = "\n\n".join(retrieved_chunks)

        # Build final prompt (retrieved context + new question)
        final_prompt = (
            "Example. Return Answer.\n\n"
            f"{retrieval_context}\n\n"
            "---\n\n"
            f"Question:\n{prompt_text}\n\n"
            "Answer:"
        )

        # Tokenize and move input to GPU
        inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Run generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_response = generated_text.strip()

        # Append result to output list
        all_results.append({
            "id": int(prompt_id),
            "prompt": prompt_text,
            "response": final_response
        })

        # Display
        print("-" * 40)
        print(f"Prompt:\n\n{prompt_text}\n")
        print(f"Answer:\n\n{final_response}\n")
        print("-" * 40)

        # Sleep between prompts
        print(f"⏳ Sleeping {SLEEP_PER_PROMPT} seconds before next prompt...\n")
        time.sleep(SLEEP_PER_PROMPT)

        torch.cuda.empty_cache()

    # Sleep after batch
    if end < total_prompts:
        print(f"⏳ Batch complete. Sleeping {SLEEP_PER_BATCH} seconds before next batch...\n")
        time.sleep(SLEEP_PER_BATCH)

# ---------------------------
# Save Final Output
# ---------------------------

output_df = pd.DataFrame(all_results)
output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"\n🎉 All prompts processed and saved to '{OUTPUT_PATH}'.")
