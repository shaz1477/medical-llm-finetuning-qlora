# medical-llm-finetuning-qlora
QLoRA fine-tuning of Meta-Llama-3.1-8B-Instruct on 10,178 medical Q&amp;A pairs using Unsloth and TRL on Kaggle T4 GPU.
# Medical LLM Fine-Tuning with QLoRA

Fine-tuning Meta-Llama-3.1-8B-Instruct on the medalpaca medical Q&A dataset
using QLoRA (Quantized Low-Rank Adaptation) via Unsloth on Kaggle T4 GPU.

## Overview

Base model frozen in INT4, only LoRA adapters trained in FP16 —
achieving full fine-tuning quality at a fraction of the memory cost.

## Results

- Dataset: 10,178 medical Q&A pairs (medalpaca/medical_meadow_medqa)
- Trainable parameters: 13.6M of 4.6B (0.29%)
- Training loss: 2.196 → 1.06 over 1,273 steps
- Hardware: Single Kaggle T4 GPU (free tier)

## Tech Stack

Unsloth · TRL (SFTTrainer) · PEFT · HuggingFace Transformers · BitsAndBytes · Kaggle

## LoRA Config

| Parameter | Value |
|---|---|
| r | 16 |
| lora_alpha | 16 |
| target_modules | q_proj, k_proj, v_proj, o_proj |
| lora_dropout | 0 |
| Training precision | FP16 |

## How to Run

1. Open the notebook on Kaggle
2. Add your HuggingFace token as a Kaggle secret (`HF_TOKEN`)
3. Enable GPU accelerator (T4)
4. Run all cells

> Note: LoRA adapters (~300MB) are saved locally during the session.
> The base model is downloaded automatically from HuggingFace.
