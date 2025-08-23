# lora-sftuner — LLM Fine-Tuning Toolkit

A modular toolkit for turning your personal archives into a high-quality Supervised Fine-Tuning (SFT) dataset — and (soon) training a LoRA adapter so a base model can “sound like you.”

* **Status:** Data pipeline ✅ • LoRA training 🚧 *coming soon*
* **Python:** 3.11–3.12
* **Platforms:** macOS / Linux (Windows via WSL)

---

## Table of Contents

1. [Features](#features)
2. [Quick Start](#quick-start)

   * [Install](#install)
   * [Clean Up](#clean-up)
3. [Configuration](#configuration)
4. [Usage: The Data Pipeline](#usage-the-data-pipeline)

   * [Step 1: Import](#step-1-import)
   * [Step 2: Unify](#step-2-unify)
   * [Step 3: Split](#step-3-split)
5. [SQL Schema Mapping Example](#sql-schema-mapping-example)
6. [Notes & Tips](#notes--tips)
7. [Roadmap](#roadmap)
8. [Contributing & License](#contributing--license)

---

## Features

* **Modular Importers**
  Incrementally ingest from multiple sources:

  * **Twitter**

    * One-time archive import.
    * Incremental sync via Twitter v2 API.
  * **Documents**

    * Scans directories for `.txt`, `.md`, `.html`, `.docx`, `.pdf`.
    * Only processes new/changed content.
  * **SQL Databases**

    * Converts forum threads or other structured content using a flexible, sidecar **YAML** schema mapping.

* **Intelligent Processing**

  * Builds **conversational dialogs** from reply chains and Twitter threads.

* **Robust Setup**

  * `setup.sh` handles Python version checks, virtualenv creation, and installs core + optional extras (e.g., PDF/DOCX readers).

* **Hierarchical Configuration**

  * Project defaults in `config.yaml`, secrets in `.env`, quick overrides via CLI flags.

* **End-to-End Data Pipeline**

  * Clear workflow from **raw data → unified dataset → train/eval splits**.

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

This will:

* Discover a compatible Python (prefers 3.11 or 3.12).
* Create a virtual environment at `./.venv`.
* Install required dependencies (CPU or CUDA variants).
* Offer to install optional document processors (`.pdf`, `.docx`, etc.).

### Clean Up

Remove the virtual environment and uninstall the package:

```bash
chmod +x clean.sh
./clean.sh
```

---

## Configuration

Before running any commands, copy the example config files:

```bash
cp config.yaml.example config.yaml
cp .env.example .env
```

* **`.env`** — secrets & user-specific settings. Fill these in:

  * `HUGGINGFACE_HUB_TOKEN`
  * `TWITTER_BEARER_TOKEN`
  * `TWITTER_USERNAME`
* **`config.yaml`** — project defaults (base model name, training presets, paths, etc.). Review and adjust as needed.

---

## Usage: The Data Pipeline

The CLI centers on three steps: **Import → Unify → Split**. All commands use the `lora-sftuner` entrypoint.

> Output artifacts are written under `dataset/`.

### Step 1: Import

Run one or more importers. Each produces a `.jsonl` file in `dataset/`.

#### Twitter Archive (one-time bulk import)

```bash
lora-sftuner twitter-import /path/to/your/twitter-archive
```

#### Twitter API (incremental sync)

Uses a state file to resume from the last fetch.

```bash
# Ensure TWITTER_USERNAME and TWITTER_BEARER_TOKEN are set in .env
lora-sftuner twitter-api-import
```

#### Documents (incremental scan)

Scans a directory and processes only new/changed files.

```bash
# If you didn't install docs extras during setup:
# pip install -e ".[docs]"

lora-sftuner docs-import /path/to/your/documents/
```

#### SQL Database

Converts threads from a SQL DB using a sidecar YAML that maps your schema.
For `my_forum.db`, create `my_forum.yaml` in the same directory.

```bash
# --nick must match your username in the database
lora-sftuner sql-import /path/to/my_forum.db --nick "YourUsername"
```

### Step 2: Unify

Merge all `.jsonl` files in `dataset/` into a single normalized dataset.

```bash
lora-sftuner unify --out dataset/unified.jsonl
```

### Step 3: Split

Create train/eval splits from the unified dataset.

```bash
lora-sftuner split-eval dataset/unified.jsonl \
  --train-out dataset/train.jsonl \
  --eval-out  dataset/eval.jsonl  \
  --eval-pct  0.05
```

You’re now ready for the next stage: **training**.

---

## SQL Schema Mapping Example

Create a YAML file matching your DB filename. Example `my_forum.yaml`:

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

Place this YAML next to `my_forum.db` so the importer can detect it automatically.

---

## Notes & Tips

* **Virtualenv activation:**
  `source .venv/bin/activate` (bash/zsh) before running commands manually.
* **Docs extras:**
  If you skipped document parsers during setup, install later with:

  ```bash
  pip install -e ".[docs]"
  ```
* **Environment variables:**
  Missing tokens will cause importer errors. Confirm your `.env` values and reload your shell or export them.
* **Idempotency:**
  Importers track state and only process new/changed content where applicable.

---

## Roadmap

* **LoRA Training Commands:** configure base model, load dataset splits, train & export an adapter.
* **More Importers:** email, chat platforms, RSS, bookmarking services.
* **Quality Tools:** deduplication, safety filters, style heuristics.

---

## Contributing & License

PRs and issues are welcome! Please open an issue for discussion before large changes.

