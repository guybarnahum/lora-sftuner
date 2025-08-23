# src/lora_sftuner/training/sft_trainer.py

import os
import math
from pathlib import Path  # <-- FIX: Import the Path object
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# --- Data Helpers ---

def _jsonl_to_dataset(path: str):
    return load_dataset("json", data_files=path, split="train")

def _apply_chat_template(example: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, str]:
    """Applies the chat template to a single example."""
    msgs = example.get("messages")
    text = ""
    if isinstance(msgs, list):
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            # Fallback for templates that might fail
            text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs)
    return {"text": text}

# --- Main Training Function ---

def run_training(config: Dict[str, Any]):
    """
    Main function to run the SFT training process.
    Accepts a configuration dictionary.
    """
    # --- FIX: Add immediate feedback for the user ---
    print("Starting training process... (Model and tokenizer loading may take a moment)")
    
    torch.manual_seed(config.get("seed", 42))

    # --- Device and Dtype Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = config.get("bf16", False)
    use_fp16 = config.get("fp16", False)
    
    if device == "cuda":
        if use_bf16 and not torch.cuda.is_bf16_supported():
            print("Warning: BF16 requested but not supported. Falling back to FP16.")
            use_bf16, use_fp16 = False, True
    else:
        use_bf16, use_fp16 = False, False # No half-precision on CPU

    print(f"Using device: {device.upper()} ({'BF16' if use_bf16 else 'FP16' if use_fp16 else 'FP32'})")

    # --- Tokenizer ---
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Datasets ---
    train_ds = _jsonl_to_dataset(config["data"]).map(
        lambda ex: _apply_chat_template(ex, tokenizer),
        desc="Applying chat template to train set"
    )
    eval_ds = None
    if config.get("eval") and Path(config["eval"]).exists():
        eval_ds = _jsonl_to_dataset(config["eval"]).map(
            lambda ex: _apply_chat_template(ex, tokenizer),
            desc="Applying chat template to eval set"
        )

    # --- Quantization (QLoRA) ---
    quantization_config = None
    if config.get("load_in_4bit"):
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        print(f"QLoRA: Using 4-bit NF4 with compute dtype: {compute_dtype}")

    # --- Model Loading ---
    torch_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        token=token,
        torch_dtype=torch_dtype,
        attn_implementation=config.get("attn_impl", "eager"),
        device_map="auto" if device == "cuda" else None,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    if config.get("gradient_checkpointing"):
        model.config.use_cache = False

    # --- LoRA Config ---
    target_modules = [t.strip() for t in config.get("target_modules", "").split(",") if t.strip()]
    peft_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # --- SFT Trainer Config ---
    sft_config = SFTConfig(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config.get("batch_size", 2),
        gradient_accumulation_steps=config.get("grad_accum", 8),
        learning_rate=config.get("lr", 2e-4),
        num_train_epochs=config.get("epochs", 1),
        logging_steps=config.get("logging_steps", 20),
        save_steps=config.get("save_steps", 200),
        seed=config.get("seed", 42),
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        dataset_text_field="text",
        max_seq_length=config.get("cutoff_len", 1024),
        packing=not config.get("no_packing", False),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        peft_config=peft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    # --- Train ---
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    print(f"\n✅ Training complete. Adapter saved to: {config['output_dir']}")
