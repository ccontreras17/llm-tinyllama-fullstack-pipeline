"""
cli_lora_inference.py

Runs interactive inference with a TinyLLaMA model fine-tuned via LoRA for Python instruction following.

This script:

- Loads the base TinyLLaMA-1.1B-Chat-v1.0 model in 8-bit precision (via BitsAndBytes)
- Injects a custom LoRA adapter trained on Python task instructions
- Formats prompts in Alpaca-style instruction-response format
- Generates responses using top-p sampling

LoRA adapter used: 'python_light'

To exit the CLI loop, type exit or quit.

"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import os

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/lora_adapters/python_light")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

print("🧠 Loading TinyLlama base model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda",
    torch_dtype=torch.float16
)

# Apply the LoRA adapter to the base model
print("🧪 Injecting LoRA adapter: python_light")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

print("✅ Model ready.\n")

while True:
    prompt = input("📝 Prompt (or type 'exit' to quit): ")

    if prompt.strip().lower() in ["exit", "quit"]:
        print("👋 Goodbye, chaos wizard.")
        break

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

    # Extract response text from generated output
    if "### Response:" in generated_text:
        final_response = generated_text.split("### Response:")[-1].strip()
    else:
        final_response = generated_text.strip()

    print("\n💬 Response:\n")
    print(final_response)
    print("-" * 60)
