"""
cli_inference.py

This script runs interactive inference using the TinyLLaMA-1.1B-Chat-v1.0 model with 8-bit quantization,
enabling efficient local testing and exploration of instruction-tuned language models.

It loads the model and tokenizer from Hugging Face, formats user input into Alpaca-style instructions,
and generates responses using standard sampling parameters. Outputs are trimmed to return only the model's
final response content.

Features:

- 8-bit quantized loading for reduced memory usage (QLoRA-style setup)
- Instruction-based prompt formatting
- Interactive command-line interface (CLI)
- Clean output parsing and response extraction

Use this script for testing and evaluating prompt behavior, or validating inference logic
prior to deployment.

To exit the loop, type exit or quit.

"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Config to load in 8-bit (to save VRAM) but run inference smartly
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

print("🧠 Loading Tiny Llama model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda",
    torch_dtype=torch.float16
)

print("✅ Model loaded. Enter your prompt below.\n")

# Interactive loop with consistent prompt formatting and response trimming
while True:
    prompt = input("📝 Prompt (or type 'exit' to quit): ")

    if prompt.strip().lower() in ["exit", "quit"]:
        print("👋 Goodbye, chaos wizard.")
        break

    # Format input the same way as your batch script
    formatted_prompt = f"### Instruction:\n{prompt.strip()}\n\n### Response:\n"

    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the actual response
    if "### Response:" in generated_text:
        final_response = generated_text.split("### Response:")[-1].strip()
    else:
        final_response = generated_text.strip()

    print("\n💬 Response:\n")
    print(final_response)
    print("-" * 60)
