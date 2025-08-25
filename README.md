# LoRA-SFTuner — LLM Fine-Tuning Toolkit

## Raw data → Unified dataset → Train/Eval split → Train LoRA → Merge/Export

**lora-sftuner** is a modular toolkit for creating personalized language models. It provides an end-to-end pipeline to import data from your archives, turn it into a high-quality Supervised Fine-Tuning (SFT) dataset, **train a LoRA adapter**, and optionally **merge/export** for inference tools like **Ollama** (via **llama.cpp** GGUF).

* **Python:** 3.11–3.12
* **Platforms:** macOS / Linux (+GPU) (Windows via WSL untested)

---

## Table of Contents

1. [Features](#features)
2. [Quick Start](#quick-start)

   * [Install](#install)
   * [Install llama.cpp (GGUF converter)](#install-llamacpp-gguf-converter)
   * [Install Ollama](#install-ollama)
   * [Clean Up](#clean-up)
3. [Configuration](#configuration)
4. [The Data Pipeline](#the-data-pipeline)

   * [Step 1: Import](#step-1-import)
   * [Step 2: Unify](#step-2-unify)
   * [Step 3: Split](#step-3-split)
5. [Training & Using Your Model](#training--using-your-model)

   * [Training](#training)
   * [Inference](#inference)
   * [Merging & Exporting for Ollama](#merging--exporting-for-ollama)
6. [SQL Schema Mapping Example](#sql-schema-mapping-example)
7. [Notes & Tips](#notes--tips)

---

## Features

* **Modular Importer System**
  Incremental importers for:

  * **Twitter** (archive import & v2 API sync)
  * **Documents** (`.txt`, `.md`, `.html`, `.docx`, `.pdf`)
  * **SQL databases** (forum/structured data via sidecar **YAML** schema)

* **Intelligent Data Processing**

  * Builds conversational dialogs from reply chains & Twitter threads
  * Cleans HTML and filters low-quality content

* **Robust Setup**

  * `setup.sh` checks Python, creates a virtualenv, and installs core + optional extras
  * Optional one-shot installers for **llama.cpp** and **Ollama**

* **Hierarchical Configuration**

  * Project defaults in `config.yaml`, secrets in `.env`, quick overrides via CLI flags

* **Complete Workflow**

  * **Raw data → Unified dataset → Train/Eval split → Train LoRA → Merge/Export (GGUF)**

---

## Quick Start

### Install

The `setup.sh` script automates everything.

```bash
# Make the script executable
chmod +x setup.sh

# Run the setup (prompts for optional components)
./setup.sh

# Auto-yes for optional prompts:
AUTO_YES=1 ./setup.sh
# or
./setup.sh --yes
```

What it does:

* Finds a compatible Python (prefers 3.11/3.12)
* Creates a virtual environment at `./.venv`
* Installs required dependencies (CPU or CUDA variants)
* Offers optional document parsers (`.pdf`, `.docx`, etc.)
* Offers to install **llama.cpp** (for GGUF) and **Ollama** (runtime)

---

### Install llama.cpp (GGUF converter)

This project uses `llama.cpp` to convert merged HF models to **GGUF** and to quantize them.
Use the dedicated script:

```bash
chmod +x setup_llama.sh
./setup_llama.sh
# Non-interactive:
AUTO_YES=1 ./setup_llama.sh
```

What it does:

* Detects your platform:

  * **Linux + NVIDIA**: builds with `GGML_CUDA=ON`
  * **macOS Apple Silicon**: builds with `GGML_METAL=ON`
  * **CPU** fallback otherwise
* Installs build deps (apt/Homebrew) and fixes common CMake deps

  * Linux includes: `pkg-config`, `libcurl4-openssl-dev`, `zlib1g-dev`, etc.
* Builds `llama-cli`, `llama-quantize`, `llama-server`
* Creates a Python venv under `~/llama.cpp/.venv` for the converter script
* Installs converter Python deps from **PyPI** only (no Torch required)
* Symlinks binaries to `~/.local/bin`
* Sets and **persists** `LLAMA_CPP_HOME` to your shell rc (e.g. `~/.bashrc`/`~/.zshrc`)

After install, open a new shell (or `source ~/.bashrc` / `source ~/.zshrc`), then:

```bash
llama-cli --version
echo "$LLAMA_CPP_HOME"
```

> If your shell can’t find `llama-cli`, ensure `~/.local/bin` is in `PATH` (the script offers to add it).

---

### Install Ollama

Ollama is optional but recommended for quick local serving of GGUF models.

```bash
chmod +x setup_ollama.sh
./setup_ollama.sh
# Non-interactive:
AUTO_YES=1 ./setup_ollama.sh
```

What it does:

* **Linux**: Installs via official script, enables & starts the systemd service
* **macOS**: Installs via Homebrew (or official script if Homebrew is missing)
* Ensures the Homebrew bin dir is in your **current** PATH and persists it for **future** shells
* Prints `ollama --version` and optional GPU hints on Linux

> To start manually (if needed): `ollama serve`
> Example test: `ollama pull llama3:8b` then `ollama run llama3:8b -q "hi"`

---

### Clean Up

**Core project** (virtualenv, package):

```bash
chmod +x clean.sh
./clean.sh
```

**llama.cpp** (binaries, venv, PATH export, env var):

```bash
chmod +x clean_llama.sh
./clean_llama.sh
# Non-interactive:
AUTO_YES=1 ./clean_llama.sh
```

**Ollama** (service + binary + optional `~/.ollama` models/cache):

```bash
chmod +x clean_ollama.sh
./clean_ollama.sh
# Non-interactive:
AUTO_YES=1 ./clean_ollama.sh
```

---

## Configuration

Copy the example config files, then fill in your details:

```bash
cp config.yaml.example config.yaml
cp .env.example .env
```

* **`.env`** — secrets & user-specific settings (required)

  * `HUGGINGFACE_HUB_TOKEN`
  * `TWITTER_BEARER_TOKEN`
  * `TWITTER_USERNAME`

* **`config.yaml`** — defaults for base model, training presets, paths, etc.
  Adjust to match your hardware and preferences.

---

## The Data Pipeline

The first stage is a three-step process: **Import → Unify → Split**.
Each importer produces a `.jsonl` file in `dataset/`.

### Step 1: Import

Run one or more importers:

* **Twitter Archive**

  ```bash
  lora-sftuner twitter-import /path/to/your/twitter-archive
  ```

* **Twitter API Sync** (incremental; uses state to resume)

  ```bash
  # Requires TWITTER_USERNAME and TWITTER_BEARER_TOKEN in .env
  lora-sftuner twitter-api-import
  ```

* **Documents** (incremental scan)

  ```bash
  # If you skipped docs extras during setup:
  # pip install -e ".[docs]"
  lora-sftuner docs-import /path/to/your/documents/
  ```

* **SQL Database** (requires sidecar YAML mapping)

  ```bash
  # --nick must match your username in the DB
  lora-sftuner sql-import /path/to/my_forum.db --nick "YourUsername"
  ```

### Step 2: Unify

Merge all generated `.jsonl` files into a single, cleaned dataset:

```bash
lora-sftuner unify --out dataset/unified.jsonl
```

### Step 3: Split

Create your final training and evaluation sets:

```bash
lora-sftuner split-eval dataset/unified.jsonl \
  --train-out dataset/train.jsonl \
  --eval-out  dataset/eval.jsonl  \
  --eval-pct  0.05
```

---

## Training & Using Your Model

Once your dataset is prepared, you can **train** a LoRA adapter, **test** it, and **merge/export** for deployment.

### Training

The trainer reads `train.jsonl` and `eval.jsonl` by default and uses your hierarchical configuration.

```bash
# Train using defaults from config.yaml
lora-sftuner train

# Train with a specific hardware preset defined in config.yaml
lora-sftuner train --preset t4_gpu

# Override settings via CLI flags
lora-sftuner train --epochs 3 --lr 1e-4
```

> The trained LoRA adapter is saved to the path configured in `config.yaml`
> (default: `out/lora-adapter`).

### Inference

Test your trained adapter:

```bash
# Uses adapter path from config.yaml by default
lora-sftuner infer "Write a short paragraph about the future of AI."

# Or specify a different adapter directory
lora-sftuner infer "Tell me a story." --adapter-dir out/another-adapter
```

### Merging & Exporting for Ollama

Merge the LoRA adapter into the base model to create a standalone, fine-tuned model.
You can also produce **GGUF** + a **Modelfile** for Ollama in one step.

```bash
# Merge the adapter into the base model
lora-sftuner merge

# Merge and convert to GGUF (example quantization)
lora-sftuner merge --gguf-quantize q4_k_m
```

Under the hood the tool will:

1. Load the base model + LoRA adapter, then **merge** and save an HF model under `out/merged` (default).
2. Use **llama.cpp**’s converter (`convert_hf_to_gguf.py`) from `LLAMA_CPP_HOME` to produce a `.gguf` file.
3. Optionally **quantize** it with `llama-quantize`.
4. Write a simple **Modelfile** (so you can `ollama create` quickly).

Run with Ollama:

```bash
ollama create my-personal-model -f out/merged/Modelfile
ollama run my-personal-model "Say something in my style."
```

---

## SQL Schema Mapping Example

Create a YAML file with the **same base name** as your DB (e.g., `my_forum.yaml` next to `my_forum.db`):

```yaml
schema_mapping:
  table_name: "msg_tbl"
  column_names:
    id: "id"
    parent_id: "parent_id"
    root_id: "root_id"
    author_nick: "nick"
    content_body: "body"
    content_title: "title"   # Optional; remove if not present
    created_at: "date"
```

---

## Notes & Tips

* **Virtualenv activation:** `source .venv/bin/activate` before running commands manually.

* **Docs extras:** If you skipped document parsers during setup, install later:

  ```bash
  pip install -e ".[docs]"
  ```

* **Idempotency:** Importers track state and only process new/changed content where applicable.

* **Environment variables:**

  * Ensure `.env` has required tokens; reload your shell after edits.
  * `setup_llama.sh` **exports** and **persists** `LLAMA_CPP_HOME` to your shell rc.
  * `setup_llama.sh` also offers to add `~/.local/bin` to your PATH for the llama binaries.

* **Linux build deps for llama.cpp:** If you’re building manually, the script already installs these, but FYI:

  ```bash
  sudo apt-get install -y git cmake build-essential g++ make python3-venv python3-pip pkg-config ccache \
                          libcurl4-openssl-dev zlib1g-dev libopenblas-dev
  ```

* **Troubleshooting (Linux):**

  * Missing `pkg-config` / `libcurl` → rerun `setup_llama.sh` or install the packages above.
  * CUDA is detected via `nvidia-smi`. `nvcc` is not required for `GGML_CUDA=ON`.
  * If you see `openblas64 not found` but `openblas found` that’s OK—OpenBLAS will still be used.

* **macOS PATH / Homebrew:**

  * `setup_ollama.sh` ensures `$(brew --prefix)/bin` is added to your PATH (current session and `~/.zshrc`).
  * If you still can’t find `ollama`, open a new terminal or `source ~/.zshrc`.

* **CLI help:** `lora-sftuner <command> --help` shows all options and overrides.

---

Happy tinkering!
