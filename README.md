# LoRA-SFTuner — LLM Fine-Tuning Toolkit
## Raw data → Unified dataset → Train/Eval split → Train LoRA → Merge/Export

**lora-sftuner** is a modular toolkit for creating personalized language models. It provides an end-to-end pipeline to import data from your archives, turn it into a high-quality Supervised Fine-Tuning (SFT) dataset, **train a LoRA adapter**, and optionally **merge/export** for inference tools like **Ollama**.

* **Python:** 3.11–3.12
* **Platforms:** macOS / Linux (+GPU)(Windows via WSL untested)

---

## Table of Contents

1. [Features](#features)
2. [Quick Start](#quick-start)

   * [Install](#install)
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

* **Hierarchical Configuration**

  * Project defaults in `config.yaml`, secrets in `.env`, quick overrides via CLI flags

* **Complete Workflow**

  * **Raw data → Unified dataset → Train/Eval split → Train LoRA → Merge/Export**

---

## Quick Start

### Install

The `setup.sh` script automates everything.

```bash
# Make the script executable
chmod +x setup.sh

# Run the setup
./setup.sh
```

What it does:

* Finds a compatible Python (prefers 3.11/3.12)
* Creates a virtual environment at `./.venv`
* Installs required dependencies (CPU/CUDA variants)
* Offers optional document parsers (`.pdf`, `.docx`, etc.)

### Clean Up

Remove the virtual environment and uninstall the package:

```bash
chmod +x clean.sh
./clean.sh
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
You can also produce GGUF + a Modelfile for Ollama in one step.

```bash
# Merge the adapter into the base model
lora-sftuner merge

# Merge and convert to GGUF (example quantization)
lora-sftuner merge --gguf-quantize q4_k_m
```

This creates a directory (default: `out/merged-model`) with merged weights and a `Modelfile`.
Run with Ollama:

```bash
ollama create my-personal-model -f out/merged-model/Modelfile
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
* **Environment variables:** Missing tokens in `.env` will cause importer errors—verify and reload your shell.
* **CLI help:** Use `lora-sftuner <command> --help` to see all options and overrides.

