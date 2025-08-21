"""
train_lora_adapter.py

This script fine-tunes a TinyLLaMA model on instruction-response pairs in Alpaca format
using parameter-efficient fine-tuning (LoRA) with 8-bit quantized weights. LoRA adaptation
is applied selectively to layers 15–20 based on activation correlation analysis.

"""

import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig


# Configuration
DATASET_PATH = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/training_data/python/basics/python_basics.jsonl")
LOG_DIR = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/logs/training_logs/python_light")
OUTPUT_ADAPTER_DIR = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/lora_adapters/python_light")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_ADAPTER_DIR, exist_ok=True)


# Load Tokenizer & Model in 8-bit
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map={"": torch.cuda.current_device()}
)

# Prepare model for 8-bit LoRA training (QLoRA-style setup)
model = prepare_model_for_kbit_training(model)


# LoRA Configuration (Layers 15–20 only)
target_modules = [
    f"model.layers.{i}.self_attn.q_proj" for i in range(15, 21)
] + [
    f"model.layers.{i}.self_attn.v_proj" for i in range(15, 21)
]

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load instruction-response dataset and reformat into Alpaca-style prompts
def alpaca_format(example):
    return {
        "text": f"### Instruction:\n{example['instruction'].strip()}\n\n### Response:\n{example['response'].strip()}"
    }

raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({
    'train': dataset['train'].map(alpaca_format),
    'validation': dataset['test'].map(alpaca_format)
})

# Tokenize dataset for causal LM with max length 128
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

dataset = DatasetDict({
    'train': dataset['train'].map(tokenize, batched=True, remove_columns=dataset['train'].column_names),
    'validation': dataset['validation'].map(tokenize, batched=True, remove_columns=dataset['validation'].column_names)
})

# Training Setup
training_args = TrainingArguments(
    output_dir=OUTPUT_ADAPTER_DIR,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir=LOG_DIR,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=None,
    eval_strategy="epoch",
    eval_steps=None,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train
print("[INFO] Starting LoRA fine-tuning...")
trainer.train()
print("[INFO]: Training complete.")

# Save final adapter
model.save_pretrained(OUTPUT_ADAPTER_DIR)
print(f"[LOG]: LoRA adapter saved to {OUTPUT_ADAPTER_DIR}")
