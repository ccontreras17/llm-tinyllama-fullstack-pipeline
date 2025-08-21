# 🧠 TinyLLaMA Inference & Interpretability

A modular, high-efficiency pipeline for batch inference, activation tracing, RAG, and LoRA-based experimentation using the TinyLLaMA-1.1B-Chat-v1.0 model. Built for research, interpretability, and dataset generation in resource-constrained environments.

### 🧠 Why Layers 15-20 Were Targeted for LoRA Fine-Tuning

Layer selection for LoRA adaptation was based on a comparative introspection study of **inter-layer activation correlation** using a quantized `TinyLLaMA-1.1B` model (8-bit, 4GB VRAM). The goal was to identify which parts of the model exhibited the most dynamic behavior under varying reasoning demands.

Using 300 prompts across three controlled sets:

- 🔍 **100 General prompts** (creative/factual tasks)
- 🐍 **100 Python prompts** (basic internal reasoning)
- 📚 **100 Python + RAG** (same Python prompts with top-1 semantic retrieval)

I measured how transformer layers correlated during generation using **mean activation trace analysis**.

#### 🧠 Observations:
- **General prompts** showed high correlation across all layers,suggesting shallow, pattern-driven generation.
- **Python-only prompts** introduced divergence in deeper layers, hinting at more logical reasoning and internal state shifts.
- **Python + RAG prompts** caused clear decorrelation in **layers 15–20**, with heavier activation signatures, indicating deeper integration between retrieved and internal context.

Based on these results, **layers 15-20 were selected for LoRA adaptation**, as they were the most responsive to contextual variation and showed non-trivial representation dynamics.

> **Layer 21 was explicitly left untouched**, as it consistently functioned as a stable output assembler rather than an active reasoning participant.

This layer-targeting strategy aims to maximize fine-tuning impact while preserving model efficiency, particularly in small-scale setups where full fine-tuning isn't feasible.

---

### 🧠 Emergent Behavior Observed in LoRA Adapter Fine-Tuning

#### Overview

During post-LoRA inference with the `python_light` adapter (targeting layers 15-20 of `TinyLLaMA-1.1B-Chat-v1.0`), several subtle emergent behaviors were observed. These suggest that the model began mimicking reflective or "reasoning-like" patterns, even without any real tool use or validation mechanism.

#### 🔍 Key Observations

**Self-Evaluation Through Explicit Checks**  
  In some cases, the model generated *actual test code* to evaluate its own outputs, defining functions to check correctness, such as confirming whether a string is a palindrome.  
  This behavior was not explicitly prompted, indicating the model may have internalized validation patterns from instruction-response examples.  

- **Redundant Output Generation**  
  In some outputs, the model produced *two* versions of the same solution, unprompted, as if it were "revising" itself.

- **Improved Prompt Format Adherence**  
  Compared to the pre-LoRA baseline, the model became more consistent in following Alpaca-style formatting, even when prompts were slightly malformed.

- **Conceptual Parroting of Reasoning**  
  It began including short justifications like:  
  `"This number is even because it is divisible by 2."`, mimicking reasoning chains seen in the training data.

- **Hallucinated Confidence**  
  Post-LoRA, the model showed increased confidence (and sometimes *overconfidence*) in its answers, often presenting incorrect information with assertiveness, likely due to reinforcement from confidently phrased training outputs.

#### 💡 Interpretation

These patterns are likely surface-level linguistic behaviors reinforced by:
- Consistent formatting in the fine-tuning data
- Targeted adaptation of deeper transformer layers
- Exposure to structured instruction-response pairs

Even without true reasoning, the adapter appears to bias the model toward producing answers that *sound* thoughtful or reflective, an effect observed in larger models exposed to chain-of-thought examples.

#### ⚠️ Limitations

- No real “checking” or reflection is occurring, the model is merely pattern-matching and parroting.
- Hallucinations remain common, especially for longer inputs or memory-heavy tasks.
- True reasoning, verification, or tool use is not present.

#### 🧠 Conclusion

Despite the model’s small size (~1B parameters) and constrained environment (4GB VRAM), the fine-tuned version demonstrated surprising linguistic alignment with “reflective” output structures. These behaviors highlight the potential of **targeted LoRA fine-tuning** to elicit structured, interpretable responses, even in lightweight, resource-conscious setups.


---

## 📦 Core Functionality

| Script | Description |
|--------|-------------|
| `batch_inference.py` | Run batched instruction-style prompt inference with 8-bit quantized TinyLLaMA. Outputs CSV of responses. |
| `batch_inference_activationtrace.py` | Capture transformer layer activations while performing inference on batches of prompts. |
| `batch_inference_rag.py` | Apply RAG (Retrieval-Augmented Generation) using semantic similarity from a reference QA dataset. |
| `batch_inference_rag_activationtrace.py` | Combine RAG with activation tracing for interpretability studies. |
| `batch_lora_inference_activationtrace.py` | Perform inference using a fine-tuned LoRA adapter and trace activations layer-wise. |
| `cli_inference.py` | Lightweight command-line interface for interactive testing with TinyLLaMA. |
| `cli_inference_rag.py` | Interactive CLI with integrated RAG using top-k semantic retrieval. |
| `cli_lora_inference.py` | Run LoRA-enhanced inference interactively via CLI for Python-instruction tasks. |
| `merge_prompts_and_responses.py` | Aligns prompt-response CSVs by ID for structured training data creation. |
| `convert_csv_pairs_to_jsonl.py` | Converts CSV pairs into JSONL for Hugging Face instruction tuning. |
| `train_lora_adapter.py` | Fine-tunes TinyLLaMA using LoRA on custom instruction data with 8-bit weights. |

---

## 🔧 Technologies Used

- [TinyLLaMA-1.1B-Chat-v1.0](https://huggingface.co/cerebras/TinyLLaMA-1.1B-Chat-v1.0)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) (8-bit quantization)
- [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685)
- [SentenceTransformers](https://www.sbert.net/)
- Retrieval Model: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- PyTorch, Hugging Face Transformers & Accelerate
- Jupyter & CLI support
- JSONL/CSV processing for dataset creation

---

## 🚀 Use Cases

- 🧪 **Prompt Evaluation** - Generate and assess LLM responses at scale
- 🔍 **Activation Analysis** - Capture transformer block activations for interpretability
- 🧷 **RAG Experiments** - Test retrieval-enhanced prompting under memory constraints
- 🧬 **LoRA Fine-tuning** - Perform efficient instruction tuning with minimal VRAM
- 🧼 **Data Preprocessing** - Convert and structure prompt/response datasets for training

---

## 🧠 Model

**Model:** `TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0`  
**Precision:** 8-bit (via BitsAndBytes)  
**Backends:** CUDA + CPU for hybrid workflows  
**LoRA Adapter:** `python_light` 

---



