# src/lora_sftuner/inference.py

import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from peft import PeftModel

# --- Model & Tokenizer Loading ---

def _load_model_and_tokenizer(model_name: str, load_in_4bit: bool, attn_impl: str, token: str):
    """Loads the base model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = None
    if load_in_4bit and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=attn_impl,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        token=token,
        trust_remote_code=True,
    )
    return model, tokenizer

# --- Inference Functions ---

def run_inference(config: Dict[str, Any]):
    """Runs streaming inference with a base model and an optional LoRA adapter."""
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    base_model, tokenizer = _load_model_and_tokenizer(
        config["model_name"], config.get("load_in_4bit", False), config.get("attn", "sdpa"), token
    )

    if config.get("adapter_dir"):
        print(f"Loading adapter from: {config['adapter_dir']}")
        model = PeftModel.from_pretrained(base_model, config["adapter_dir"], is_trainable=False)
    else:
        model = base_model
    
    model.eval()

    messages = [{"role": "user", "content": config["prompt"]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=config.get("max_new_tokens", 512),
        do_sample=True,
        temperature=config.get("temperature", 0.8),
        top_p=config.get("top_p", 0.95),
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    base_model_name = model.config._name_or_path
    # If it's a PeftModel, the actual base model is nested
    if hasattr(model, "base_model"):
        base_model_name = model.base_model.model.config._name_or_path
    
    print(f"Base Model: {base_model_name}")
    if config.get("adapter_dir"):
        print(f"Adapter:    {config['adapter_dir']}")

    for new_text in streamer:
        print(new_text, end="", flush=True)
    
# --- Merging and Conversion ---

def merge_and_export(config: Dict[str, Any]):
    """Merges a LoRA adapter into the base model and saves it."""
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    print(f"Loading base model: {config['model_name']}")
    model, tokenizer = _load_model_and_tokenizer(
        config["model_name"], load_in_4bit=False, attn_impl="eager", token=token
    )
    
    print(f"Loading adapter: {config['adapter_dir']}")
    model = PeftModel.from_pretrained(model, config["adapter_dir"])
    
    print("Merging adapter...")
    merged_model = model.merge_and_unload()
    
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    if config.get("gguf_quantize"):
        _export_to_gguf(output_dir, config["gguf_quantize"])

def _export_to_gguf(model_path: Path, quant_type: str):
    """Exports a Hugging Face model to GGUF format for Ollama."""
    try:
        from llama_cpp.llama_model import LlamaModel
        
        gguf_path = model_path / f"ggml-model-{quant_type}.gguf"
        print(f"Converting to GGUF ({quant_type})...")
        
        LlamaModel.convert_hf_to_gguf(
            model_path,
            outfile=gguf_path,
            outtype=quant_type,
        )
        print(f"✅ GGUF model saved to: {gguf_path}")
        _create_ollama_modelfile(model_path, gguf_path.name)

    except ImportError:
        print("\nWarning: `llama-cpp-python` is not installed. Skipping GGUF conversion.")
        print("To enable Ollama export, run: pip install llama-cpp-python")
    except Exception as e:
        print(f"\nError during GGUF conversion: {e}")

def _create_ollama_modelfile(model_path: Path, gguf_filename: str):
    """Creates a basic Modelfile for Ollama."""
    modelfile_content = f"""
FROM ./{gguf_filename}
TEMPLATE "{{ .System }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"
"""
    modelfile_path = model_path / "Modelfile"
    modelfile_path.write_text(modelfile_content.strip())
    print(f"✅ Ollama Modelfile created at: {modelfile_path}")
    print(f"\nTo run with Ollama, use: ollama create {model_path.name} -f {modelfile_path}")


