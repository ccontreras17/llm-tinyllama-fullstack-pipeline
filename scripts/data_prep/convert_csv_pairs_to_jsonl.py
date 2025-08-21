"""
convert_csv_pairs_to_jsonl.py

This script converts merged prompt-response CSV files into a single JSONL file
formatted for instruction tuning (Alpaca-style).

Checks for missing files and malformed data, and logs the process clearly.
Intended as a preprocessing step for fine-tuning language models with Hugging Face.

"""

import pandas as pd
import os
import json

#Configuration

# Directory containing merged prompt-response CSV files
merged_dir = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/training_data/python/basics/merged")

# Path to output JSONL file
output_path = os.path.expanduser("~/Chaos-Projects/Python/Wednesday/training_data/python/basics/python_basics.jsonl")

num_files = 20 # Number of file pairs to process

entries = []

for i in range(1, num_files + 1):
    csv_path = os.path.join(merged_dir, f"prompt_qa_pairs{i}.csv")
    if not os.path.isfile(csv_path):
        print(f"[WARNING] Skipping missing file: {csv_path}")
        continue

    try:
        df = pd.read_csv(csv_path)
        if "prompt" not in df.columns or "response" not in df.columns:
            print(f"[ERROR] Missing 'prompt' or 'response' column in {csv_path}")
            continue

        for _, row in df.iterrows():
            entry = {
                "instruction": row["prompt"].strip(),
                "response": row["response"].strip()
            }
            entries.append(entry)

        print(f"[INFO] Processed {csv_path}, {len(df)} entries added.")

    except Exception as e:
        print(f"[ERROR] Failed to process {csv_path}: {e}")

# Save to JSONL
with open(output_path, "w", encoding="utf-8") as f:
    for entry in entries:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

print(f"\n[DONE] Merged {len(entries)} examples into: {output_path}")
