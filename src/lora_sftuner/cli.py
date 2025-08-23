import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer
import yaml
from dotenv import load_dotenv

# Import all the project's modules
from .importers import docs_importer, sql_importer, twitter_api_importer, twitter_importer
from . import sft_unify_and_split

# --- Configuration ---
APP_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(APP_DIR / ".env")

VENV_DIR = APP_DIR / os.getenv("VENV_DIR", ".venv")
CONFIG_FILE = APP_DIR / "config.yaml"
DATASET_DIR = APP_DIR / "dataset"
STATE_DIR = APP_DIR / "state"

app = typer.Typer(
    name="lora-sftuner",
    help="A toolkit for creating personalized language models by fine-tuning on personal data archives.",
    add_completion=False,
)

# --- Config Loading ---
def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            try: return yaml.safe_load(f)
            except yaml.YAMLError as e:
                typer.secho(f"Warning: Could not parse config.yaml. Error: {e}", fg=typer.colors.YELLOW)
    return {}
config = load_config()

# --- Helper Functions ---
def _ensure_venv():
    if sys.prefix == sys.base_prefix:
        typer.secho(f"Error: Not in a virtual environment. Please run 'source {VENV_DIR.name}/bin/activate' first.", fg=typer.colors.RED)
        raise typer.Exit(1)

# --- CLI Commands ---

@app.command("twitter-import")
def twitter_import_cmd(
    archive_path: Path = typer.Argument(..., help="Path to unzipped archive folder or a .js file.", exists=True),
    out: Path = typer.Option(DATASET_DIR / "twitter_archive.jsonl", help="Output JSONL file path."),
    eval_pct: float = typer.Option(0.0, help="Fraction for eval split. 0 to disable."),
    include_replies: bool = typer.Option(True, help="Include replies."),
    no_quotes: bool = typer.Option(False, help="Exclude quote tweets."),
    dialog: bool = typer.Option(True, help="Build dialogs from reply chains."),
    role_assistant: str = typer.Option("model", help="Role name for your replies ('model' or 'assistant')."),
):
    """Imports and converts a one-time Twitter/X archive."""
    twitter_importer.process_archive(
        archive_path=archive_path,
        out_path=out,
        eval_pct=eval_pct,
        include_replies=include_replies,
        no_quotes=no_quotes,
        dialog=dialog,
        role_assistant=role_assistant,
    )

@app.command("twitter-api-import")
def twitter_api_import_cmd(
    username: str = typer.Option(lambda: os.getenv("TWITTER_USERNAME"), help="Twitter username to sync."),
    bearer_token: str = typer.Option(lambda: os.getenv("TWITTER_BEARER_TOKEN"), help="Twitter API v2 Bearer token.", hide_input=True),
    out: Path = typer.Option(DATASET_DIR / "twitter_api.jsonl", help="Output JSONL file to append to."),
    state: Path = typer.Option(STATE_DIR / "twitter_api_sync.json", help="Path to the state file."),
    min_len: int = typer.Option(10, help="Minimum tweet text length."),
    exclude_sources: str = typer.Option("", help="Comma-separated app names to skip."),
    include_replies: bool = typer.Option(True, help="Include replies."),
    no_quotes: bool = typer.Option(False, help="Exclude quote tweets."),
    role_assistant: str = typer.Option("model", help="Role name for your replies ('model' or 'assistant')."), # <-- FIX
):
    """Incrementally syncs new tweets from the Twitter API."""
    exclude_sources_set = {s.strip() for s in exclude_sources.split(',') if s.strip()}
    twitter_api_importer.sync_tweets(
        username=username,
        bearer_token=bearer_token,
        out_path=out,
        state_path=state,
        min_len=min_len,
        exclude_sources=exclude_sources_set,
        include_replies=include_replies,
        no_quotes=no_quotes,
        role_assistant=role_assistant, # <-- FIX
    )

@app.command("sql-import")
def sql_import_cmd(
    input_path: Path = typer.Argument(..., help=".sql dump or .db/.sqlite file.", exists=True),
    out: Path = typer.Option(DATASET_DIR / "sql_threads.jsonl", help="Output JSONL file path."),
    nick: str = typer.Option(..., help="Your author nickname in the database."),
    max_context: int = typer.Option(8, help="Max prior turns to include in context."),
    strip_self_context: bool = typer.Option(False, help="Drop your own earlier replies from context."),
    role_assistant: str = typer.Option("model", help="Role name for your replies ('model' or 'assistant').")
):
    """Imports and converts threads from a SQL database with a sidecar .yaml config."""
    sql_importer.process_database(
        db_path=input_path,
        out_path=out,
        user_nick=nick,
        max_context=max_context,
        strip_self_context=strip_self_context,
        role_assistant=role_assistant,
    )

@app.command("docs-import")
def docs_import_cmd(
    path: Path = typer.Argument(..., help="File or directory of documents to process.", exists=True),
    out: Path = typer.Option(DATASET_DIR / "docs.jsonl", help="Output JSONL file (will be appended to)."),
    state: Path = typer.Option(STATE_DIR / "docs_sync.json", help="Path to the state file."),
    min_chars: int = typer.Option(80, help="Minimum character length for a text chunk."),
    max_chars: int = typer.Option(1200, help="Maximum character length for a text chunk."),
    tag_lang: str = typer.Option("", help="Language tag to add to each record (e.g., 'en', 'he')."),
    delete_missing: bool = typer.Option(False, help="Prune state entries for files that no longer exist."),
):
    """Incrementally ingests documents (.txt, .md, .html, .docx, .pdf) into SFT JSONL."""
    docs_importer.process_documents(
        root_path=path,
        out_path=out,
        state_path=state,
        min_chars=min_chars,
        max_chars=max_chars,
        tag_lang=tag_lang,
        delete_missing=delete_missing,
    )

@app.command()
def unify(
    inputs: List[Path] = typer.Option(None, "--in", help="Input JSONL file (can be specified multiple times). If empty, uses all .jsonl in dataset/."),
    out: Path = typer.Option(DATASET_DIR / "unified.jsonl", help="Unified output JSONL file path."),
    shuffle: bool = typer.Option(True, help="Shuffle the unified dataset."),
    seed: int = typer.Option(42, help="Random seed for shuffling."),
    keep: str = typer.Option("messages,created_at", help="Comma-separated list of keys to keep."),
):
    """Unifies and normalizes multiple JSONL datasets into one."""
    input_paths = inputs
    if not input_paths:
        input_paths = list(DATASET_DIR.glob("*.jsonl"))
        input_paths = [p for p in input_paths if p.resolve() != out.resolve() and "_eval" not in p.stem and "train" not in p.stem]

    if not input_paths:
        typer.secho("No input files found to unify.", fg=typer.colors.YELLOW)
        raise typer.Exit()
        
    typer.echo("Found files to unify:")
    for p in input_paths:
        typer.echo(f"  - {p.name}")
        
    keep_keys = [k.strip() for k in keep.split(',')]
    
    sft_unify_and_split.unify_datasets(
        input_paths=input_paths,
        output_path=out,
        shuffle=shuffle,
        seed=seed,
        keep_keys=keep_keys,
    )

@app.command("split-eval")
def split_eval(
    in_file: Path = typer.Argument(..., help="The unified JSONL file to split.", exists=True),
    train_out: Path = typer.Option(DATASET_DIR / "train.jsonl", help="Output file for the training set."),
    eval_out: Path = typer.Option(DATASET_DIR / "eval.jsonl", help="Output file for the evaluation set."),
    eval_pct: float = typer.Option(0.05, help="Fraction of data to use for the evaluation set (e.g., 0.05 for 5%)."),
    seed: int = typer.Option(42, help="Random seed for shuffling."),
):
    """Splits a unified dataset into training and evaluation sets."""
    sft_unify_and_split.split_dataset(
        input_path=in_file,
        train_path=train_out,
        eval_path=eval_out,
        eval_pct=eval_pct,
        seed=seed,
    )

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train(
    ctx: typer.Context,
    preset: Optional[str] = typer.Option(None, help="Hardware preset to use from config.yaml."),
):
    """Fine-tunes the language model."""
    typer.echo("Running training...")
    # Placeholder for actual training logic

if __name__ == "__main__":
    app()

