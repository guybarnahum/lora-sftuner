# src/lora_sftuner/inference.py

import os
import sys
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import time  # retained (used earlier for consistency if needed)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from peft import PeftModel

# --- FIX: Suppress PyTorch Dynamo errors ---
# This helps prevent crashes with incompatible attention implementations.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
try:
    import torch._dynamo as dynamo  # type: ignore[attr-defined]
    dynamo.config.suppress_errors = True  # type: ignore[attr-defined]
except Exception:
    pass


# ==============================
# Model & Tokenizer Loading
# ==============================

def _load_model_and_tokenizer(model_name: str, load_in_4bit: bool, attn_impl: str, token: Optional[str]):
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


# ==============================
# Streaming Inference
# ==============================

def run_inference(config: Dict[str, Any]):
    """Runs streaming inference with a base model and an optional LoRA adapter."""
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")

    # --- Default to 'eager' attention for better stability ---
    base_model, tokenizer = _load_model_and_tokenizer(
        config["model_name"], config.get("load_in_4bit", False), config.get("attn", "eager"), token
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

    # --- Print model and adapter info before streaming ---
    print("\n--- Configuration ---")
    base_model_name = getattr(model.config, "_name_or_path", None) or str(config["model_name"])
    # If wrapped by PEFT we may find the inner base name here:
    try:
        base_model_name = model.base_model.model.config._name_or_path  # type: ignore[attr-defined]
    except Exception:
        pass

    print(f"Base Model: {base_model_name}")
    if config.get("adapter_dir"):
        print(f"Adapter:    {config['adapter_dir']}")
    print("----------------------\n")

    print("--- Model Response ---")
    for new_text in streamer:
        print(new_text, end="", flush=True)
    print("\n----------------------")


# ==============================
# llama.cpp Tooling Discovery
# ==============================

def _which(exe: str) -> Optional[str]:
    """Shutil.which with a tiny wrapper to keep type checkers happy."""
    from shutil import which as _which
    return _which(exe)

def _resolve_repo_from_binary(bin_name: str) -> Optional[Path]:
    """
    Given a llama binary name (e.g., 'llama-quantize' or 'llama-cli'),
    try to locate it on PATH, resolve symlinks, and walk up to the repo root:
      .../llama.cpp/build/bin/<bin_name> -> repo root = .../llama.cpp
    """
    p = _which(bin_name)
    if not p:
        return None
    real = Path(os.path.realpath(p))
    # Expect .../llama.cpp/build/bin/<bin>
    try:
        if real.parent.name == "bin" and real.parent.parent.name == "build":
            repo = real.parent.parent.parent
            if (repo / "convert_hf_to_gguf.py").is_file():
                return repo
    except Exception:
        pass
    return None

def _llama_paths() -> Dict[str, Path]:
    """
    Locate llama.cpp repo, its venv python, converter script, and binaries.
    Order of precedence:
      1) $LLAMA_CPP_HOME (if valid)
      2) Derive from llama-quantize on PATH
      3) Derive from llama-cli on PATH
    Falls back to current Python if repo venv python not found.
    """
    # 1) Env override
    env_home = os.environ.get("LLAMA_CPP_HOME")
    if env_home:
        repo = Path(env_home).expanduser().resolve()
        if (repo / "convert_hf_to_gguf.py").is_file():
            venv_py = repo / ".venv" / "bin" / "python"
            return {
                "llama_dir": repo,
                "venv_python": venv_py if venv_py.is_file() else Path(sys.executable),
                "converter": repo / "convert_hf_to_gguf.py",
                "quantize": repo / "build" / "bin" / "llama-quantize",
                "cli": repo / "build" / "bin" / "llama-cli",
            }

    # 2) Derive from PATHed binaries
    for name in ("llama-quantize", "llama-cli"):
        repo = _resolve_repo_from_binary(name)
        if repo:
            venv_py = repo / ".venv" / "bin" / "python"
            return {
                "llama_dir": repo,
                "venv_python": venv_py if venv_py.is_file() else Path(sys.executable),
                "converter": repo / "convert_hf_to_gguf.py",
                "quantize": repo / "build" / "bin" / "llama-quantize",
                "cli": repo / "build" / "bin" / "llama-cli",
            }

    # 3) Last-resort: return placeholders so _check_llama_tools can error clearly
    return {
        "llama_dir": Path(env_home or ""),
        "venv_python": Path(sys.executable),
        "converter": Path("convert_hf_to_gguf.py"),
        "quantize": Path(_which("llama-quantize") or ""),
        "cli": Path(_which("llama-cli") or ""),
    }

def _check_llama_tools(paths: Dict[str, Path]) -> None:
    missing = []
    if not paths["converter"].is_file():
        missing.append(f"converter script: {paths['converter']}")
    if not paths["quantize"].is_file():
        missing.append(f"llama-quantize binary: {paths['quantize']}")
    if missing:
        repo_hint = paths.get("llama_dir", "")
        raise RuntimeError(
            "llama.cpp tools not found.\n  Missing:\n    - "
            + "\n    - ".join(missing)
            + "\n\nHints:\n"
              "  • Ensure your llama.cpp build created symlinks in ~/.local/bin (in PATH), e.g. 'llama-quantize'.\n"
              "  • Or export LLAMA_CPP_HOME=/path/to/llama.cpp so we can find convert_hf_to_gguf.py.\n"
              f"  • Current guessed repo root: {repo_hint}"
        )

def _run(cmd: list, cwd: Optional[Path] = None) -> None:
    """Run a subprocess with output attached to this TTY; raise on failure."""
    print("→", " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(str(c) for c in cmd)}")


# ==============================
# Merging and Conversion
# ==============================

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
    """
    Export to GGUF using llama.cpp tools:
      1) convert_hf_to_gguf.py → F16 base .gguf
      2) llama-quantize → desired quant (e.g., Q4_K_M)
    Looks up llama.cpp via LLAMA_CPP_HOME or PATHed binaries.
    """
    paths = _llama_paths()
    _check_llama_tools(paths)

    # 1) Convert HF → GGUF (F16)
    f16_out = model_path / "ggml-model-f16.gguf"
    print(f"Converting to GGUF (f16) using: {paths['converter']}")
    _run([
        str(paths["venv_python"]),
        str(paths["converter"]),
        str(model_path),
        "--outfile", str(f16_out),
        "--outtype", "f16",
    ])

    # 2) Quantize
    q_arg = quant_type.strip().upper()  # e.g., q4_k_m → Q4_K_M
    quant_out = model_path / f"ggml-model-{q_arg}.gguf"
    print(f"Quantizing {f16_out.name} → {quant_out.name} ({q_arg}) using: {paths['quantize']}")
    _run([
        str(paths["quantize"]),
        str(f16_out),
        str(quant_out),
        q_arg,
    ])

    print(f"✅ GGUF model saved to: {quant_out}")
    _create_ollama_modelfile(model_path, quant_out.name)


def _create_ollama_modelfile(model_path: Path, gguf_filename: str):
    """Creates a basic Modelfile for Ollama."""
    modelfile_content = f"""FROM ./{gguf_filename}
TEMPLATE "{{{{ .System }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"
"""
    modelfile_path = model_path / "Modelfile"
    modelfile_path.write_text(modelfile_content.strip() + "\n")
    print(f"✅ Ollama Modelfile created at: {modelfile_path}")
    print(f"To create an Ollama model: ollama create {model_path.name} -f {modelfile_path}")
