"""
cli_inference_rag.py

This script performs Retrieval-Augmented Generation (RAG) using a quantized TinyLLaMA-1.1B model
combined with semantic similarity search via SentenceTransformers.

Overview:

- Loads TinyLLaMA-1.1B-Chat-v1.0 using 8-bit quantization (via BitsAndBytes) for efficient inference.
- Loads a retrieval dataset of question-answer pairs and encodes them using the
  `all-MiniLM-L6-v2` embedding model from SentenceTransformers.
- Embeddings are computed and stored on CPU; only the top-k (k=1) most similar example is retrieved
  and used as few-shot context during generation.
- The final prompt includes this retrieved example, followed by the user’s input instruction.

Note:
Due to TinyLLaMA's limited attention capacity, only the single most relevant example (k=1)
is included in the prompt to prevent truncation and preserve output quality.

Features:

- Retrieval using `all-MiniLM-L6-v2` (CPU-based)
- 8-bit quantized TinyLLaMA for low-memory environments (GPU-based)
- Alpaca-style instruction formatting for in-context examples
- Interactive command-line interface with real-time response generation

To exit: type exit or quit.

"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
import torch
import json
import os


# Load Retrieval Examples
RAG_JSON = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/rag/200_python_basics.json")

print("📖 Loading retrieval examples...")
with open(RAG_JSON, "r") as f:
    examples = json.load(f)

# Load embedding model for semantic similarity search
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Encode all stored instruction examples into embedding vectors
example_texts = [ex["question"] for ex in examples]
example_embeddings = embedder.encode(
    example_texts,
    convert_to_tensor=True,
    batch_size=32,
    show_progress_bar=True
)

print(f"✅ Loaded and embedded {len(examples)} retrieval examples.\n")

# Load Language Model (8-bit)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

print("🧠 Loading Tiny Llama model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda",
    torch_dtype=torch.float16
)
print("✅ Model loaded. Enter your prompt below.\n")

# Interactive RAG Inference Loop

while True:
    prompt = input("📝 Prompt (or type 'exit' to quit): ")

    if prompt.strip().lower() in ["exit", "quit"]:
        print("👋 Goodbye, chaos wizard.")
        break


    # Semantic Retrieval
    prompt_embedding = embedder.encode(prompt, convert_to_tensor=True)
    similarities = util.cos_sim(prompt_embedding, example_embeddings)[0]
    top_indices = torch.topk(similarities, k=1).indices

    retrieved_chunks = []
    for idx in top_indices:
        ex = examples[idx]
        retrieved_chunks.append(
            f"###Example\n### Instruction:\n{ex['question']}\n\n### Response:\n{ex['answer']}"
        )

    retrieval_context = "\n\n".join(retrieved_chunks)


    # Format Final Prompt
    final_prompt = (
        f"{retrieval_context}\n\n"
        f"### Instruction:\n{prompt.strip()}\n\n"
        f"### Response:\n"
    )

    inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}


    # Generate Model Response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract Model Output Only
    if "### Response:" in decoded:
        final_response = decoded.split("### Response:")[-1]
        final_response = final_response.split("### Instruction:")[0].strip()
    else:
        final_response = decoded.strip()

    print("\n💬 Response:\n")
    print(final_response)
    print("-" * 60)

    torch.cuda.empty_cache()
