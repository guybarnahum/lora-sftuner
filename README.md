lora-sftuner: Your Personal AI Finetuning Toolkitlora-sftuner is a comprehensive toolkit for creating personalized language models. It provides a complete pipeline to import data from your personal archives, process it into a high-quality Supervised Finetuning (SFT) dataset, and (soon) train a LoRA adapter to make a base model sound just like you.This project is designed to be modular and extensible, allowing you to easily add new data sources and build a rich, multi-faceted dataset that truly reflects your unique style and knowledge.FeaturesModular Importer System: Easily ingest data from various sources with dedicated, incremental importers.Twitter: Supports both one-time archive imports and incremental syncs from the v2 API.Documents: Scans directories for .txt, .md, .html, .docx, and .pdf files, only processing new or changed content.SQL Databases: Converts forum threads or other structured data using a flexible schema mapping defined in a sidecar YAML file.Intelligent Data Processing: Automatically creates conversational dialogs from reply chains and Twitter threads.Robust Setup: A simple setup script handles Python version checking, virtual environment creation, and installation of core and optional dependencies.Hierarchical Configuration: Manage settings easily with a combination of a global config.yaml for project defaults, a .env file for secrets, and command-line flags for overrides.Complete Data Pipeline: A clear, step-by-step workflow from raw data to a unified, split dataset ready for training.1. Setup and InstallationGetting started with lora-sftuner is handled by two simple scripts.InstallationThe setup.sh script automates the entire installation process.# Make the script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
This script will:Find a compatible Python version (3.11 or 3.12 is preferred).Create a virtual environment in ./.venv.Install all required core dependencies for your platform (CPU or CUDA).Ask you if you want to install optional dependencies for document processing (.pdf, .docx, etc.).Cleaning UpTo completely remove the virtual environment and uninstall the package, use the clean.sh script.chmod +x clean.sh
./clean.sh
2. ConfigurationBefore running any commands, you should configure your project by copying the example files:cp config.yaml.example config.yaml
cp .env.example .env
.env: This file is for your secrets (like HUGGINGFACE_HUB_TOKEN and TWITTER_BEARER_TOKEN) and user-specific settings (TWITTER_USERNAME). You must fill this out.config.yaml: This file contains all the project defaults, such as the base model name, hardware presets for training, and default paths. You can review and adjust these settings as needed.3. Usage: The Data PipelineThe core workflow is a three-step process: Import, Unify, and Split. All commands are run through the lora-sftuner CLI.Step 1: Import Your DataRun one or more of the following commands to process your raw data archives. Each command will generate a .jsonl file in the dataset/ directory.Twitter Archive (One-Time)Use this for your initial bulk import from a downloaded Twitter archive.lora-sftuner twitter-import /path/to/your/twitter-archive
Twitter API (Incremental Sync)Use this to periodically fetch new tweets. It uses a state file to remember where it left off.# Make sure TWITTER_USERNAME and TWITTER_BEARER_TOKEN are set in your .env file
lora-sftuner twitter-api-import
Documents (Incremental)Scans a directory for documents and only processes new or changed files.# First, ensure you have installed the optional dependencies during setup.
# If not, run: pip install -e ".[docs]"

lora-sftuner docs-import /path/to/your/documents/
SQL DatabaseConverts threads from a SQL database. This command requires a sidecar .yaml file that maps your database schema. For an input file named my_forum.db, you must create a my_forum.yaml in the same directory.# The --nick must match your username in the database
lora-sftuner sql-import /path/to/my_forum.db --nick "YourUsername"
SQL Schema ConfigurationThe importer needs to know how to interpret your database. Create a .yaml file with the same name as your database file (e.g., my_forum.yaml) and define the schema mapping:# Example: my_forum.yaml
schema_mapping:
  table_name: "msg_tbl"
  column_names:
    id: "id"
    parent_id: "parent_id"
    root_id: "root_id"
    author_nick: "nick"
    content_body: "body"
    content_title: "title" # Optional: remove if your table has no title column
    created_at: "date"
Step 2: Unify DatasetsAfter importing from one or more sources, combine them into a single, normalized dataset.# This command finds all .jsonl files in dataset/ and merges them
lora-sftuner unify --out dataset/unified.jsonl
Step 3: Split for TrainingFinally, split your unified dataset into training and evaluation sets.lora-sftuner split-eval dataset/unified.jsonl \
  --train-out dataset/train.jsonl \
  --eval-out dataset/eval.jsonl \
  --eval-pct 0.05
You are now ready for the next stage: training
