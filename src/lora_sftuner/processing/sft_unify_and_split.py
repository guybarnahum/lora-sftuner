# src/lora_sftuner/processing/sft_unify_and_split.py

import hashlib
import html
import json
import pathlib
import random
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm

# --- Text Normalization and Hashing ---

def _norm_text(t: str) -> str:
    """Cleans text by removing HTML tags, unescaping entities, and normalizing whitespace."""
    if not t:
        return ""
    # Remove HTML tags
    t = re.sub(r"<[^>]+>", " ", t)
    # Unescape HTML entities
    t = html.unescape(t)
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _hash_for_dedup(messages: List[Dict[str, str]]) -> str:
    """Creates a hash from the last assistant message for deduplication."""
    for m in reversed(messages):
        if m.get("role", "").lower() in ("assistant", "model"):
            return hashlib.sha256(_norm_text(m.get("content", "")).lower().encode()).hexdigest()
    # Fallback: hash the whole dialog
    stitched = " ".join(_norm_text(m.get("content", "")) for m in messages)
    return hashlib.sha256(stitched.lower().encode()).hexdigest()

# --- Role and Message Normalization ---

def _map_role(role: str) -> str:
    r = (role or "").strip().lower()
    if r in ("assistant", "model"): return "assistant"
    if r in ("user", "system"): return r
    return "user"

def _enforce_alternation(cleaned: List[Dict[str, str]]) -> Optional[List[Dict[str, str]]]:
    """
    Enforces a [system?, user, assistant, user, ...] structure.
    This version is more careful about preserving multi-turn conversations.
    """
    if not cleaned: return None

    # 1. Merge true consecutive messages of the same role
    merged = []
    for msg in cleaned:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] = (merged[-1]["content"] + "\n\n" + msg["content"]).strip()
        else:
            merged.append(msg)
    
    # 2. Ensure the conversation starts with a non-assistant message
    if not merged or merged[0]["role"] == "assistant":
        # Insert a synthetic prompt to fix conversations starting with the assistant
        merged.insert(0, {"role": "user", "content": "..."})

    # 3. Ensure the conversation ends with an assistant message
    if merged[-1]["role"] != "assistant":
        merged.pop() # Drop trailing user/system message

    # 4. Final check for strict user/assistant alternation
    final_chat = []
    if merged and merged[0]["role"] == "system":
        final_chat.append(merged.pop(0))
        
    if not merged or len(merged) % 2 != 0: # Must be an even number of user/assistant turns
        return None
        
    for i, msg in enumerate(merged):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if msg["role"] != expected_role:
            return None # The sequence is broken
        final_chat.append(msg)

    return final_chat if len(final_chat) >= 2 else None


def _normalize_row(row: Dict[str, Any], keep_keys: List[str]) -> Optional[Dict[str, Any]]:
    """Normalizes a single JSONL row to the required SFT format."""
    msgs = row.get("messages")
    if not isinstance(msgs, list): return None

    cleaned_msgs = [{"role": _map_role(m.get("role")), "content": _norm_text(m.get("content"))} for m in msgs if m.get("content")]
    
    # Filter out retweets by checking the assistant's final message
    if cleaned_msgs and cleaned_msgs[-1]["role"] == "assistant":
        if cleaned_msgs[-1]["content"].strip().lower().startswith("rt"):
            return None

    fixed_msgs = _enforce_alternation(cleaned_msgs)
    if not fixed_msgs: return None

    norm_row = {"messages": fixed_msgs}
    for key in keep_keys:
        if key != "messages" and key in row:
            norm_row[key] = str(row[key]) if isinstance(row[key], datetime) else row[key]
            
    return norm_row

# --- Data Quality Analysis ---

def _is_generic_prompt(prompt: str) -> bool:
    """Determines if a user prompt is a generic placeholder."""
    p_lower = prompt.lower()
    return p_lower == "..." or p_lower.startswith("write a")

def _analyze_dataset_quality(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyzes the unified dataset and returns a dictionary of metrics."""
    if not rows: return {}

    source_counts = Counter(row.get("source_file", "unknown") for row in rows)
    
    turn_counts = Counter()
    generic_prompts = 0
    total_prompt_len = 0
    total_response_len = 0

    for row in rows:
        messages = row.get("messages", [])
        user_turns = [m for m in messages if m["role"] == "user"]
        
        turn_counts[len(user_turns)] += 1
            
        if user_turns and _is_generic_prompt(user_turns[-1]["content"]):
            generic_prompts += 1
            
        total_prompt_len += sum(len(m["content"]) for m in user_turns)
        total_response_len += sum(len(m["content"]) for m in messages if m["role"] == "assistant")

    total = len(rows)
    single_turn_count = turn_counts.get(1, 0)
    multi_turn_count = total - single_turn_count
    
    return {
        "total_examples": total,
        "source_composition": {k: v / total for k, v in source_counts.items()},
        "single_turn_pct": single_turn_count / total,
        "multi_turn_pct": multi_turn_count / total,
        "generic_prompt_pct": generic_prompts / total,
        "avg_prompt_len": total_prompt_len / total if total > 0 else 0,
        "avg_response_len": total_response_len / total if total > 0 else 0,
    }

def _print_quality_report(metrics: Dict[str, Any]):
    """Prints the formatted data quality report."""
    if not metrics: return

    print("\n--- 📊 Data Quality Report ---")
    print("Source Composition:")
    for source, pct in sorted(metrics["source_composition"].items()):
        print(f"  - {source:<25}: {pct: >7.1%}")

    print("\nConversational Quality:")
    print(f"  - Single-Turn (Q&A):   {metrics['single_turn_pct']: >7.1%}")
    print(f"  - Multi-Turn (Dialog): {metrics['multi_turn_pct']: >7.1%}")
    print(f"  - Meaningful Prompts:  {(1.0 - metrics['generic_prompt_pct']): >7.1%}")
    print(f"  - Generic Prompts:     {metrics['generic_prompt_pct']: >7.1%}")

    print("\nLength Statistics (characters):")
    print(f"  - Avg. Prompt Length:  {metrics['avg_prompt_len']:.0f}")
    print(f"  - Avg. Response Length:{metrics['avg_response_len']:.0f}")

    if metrics["generic_prompt_pct"] > 0.5:
        print("\n⚠️  Warnings:")
        print(f"  - High Generic Prompt Ratio ({metrics['generic_prompt_pct']:.1%}): A majority of your data consists of")
        print("    standalone statements with a generic prompt. This can weaken the model's")
        print("    ability to follow specific instructions.")
        print("    Suggestion: Use the --drop-generic-prompts flag or curate more direct Q&A examples.")
    print("--------------------------------")


# --- File I/O ---

def _iter_jsonl(paths: List[pathlib.Path]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        row["source_file"] = p.name
                        yield row
                    except json.JSONDecodeError:
                        continue

def _save_jsonl(p: pathlib.Path, rows: List[Dict[str, Any]]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ex in rows:
            ex.pop("source_file", None)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

# --- Main Unify Function ---

def unify_datasets(
    input_paths: List[pathlib.Path],
    output_path: pathlib.Path,
    shuffle: bool,
    seed: int,
    keep_keys: List[str],
    drop_generic_prompts: bool,
):
    """Unifies multiple JSONL datasets into a single, normalized file."""
    all_rows = list(tqdm(_iter_jsonl(input_paths), desc="Reading files"))
    
    kept, seen_hashes = [], set()
    for row in tqdm(all_rows, desc="Normalizing and deduplicating"):
        norm_row = _normalize_row(row, keep_keys + ["source_file"])
        if not norm_row: continue
        
        # Optionally drop rows with generic prompts
        if drop_generic_prompts:
            user_turns = [m for m in norm_row["messages"] if m["role"] == "user"]
            if user_turns and _is_generic_prompt(user_turns[-1]["content"]):
                continue

        h = _hash_for_dedup(norm_row["messages"])
        if h in seen_hashes: continue
        
        seen_hashes.add(h)
        kept.append(norm_row)

    if shuffle:
        random.Random(seed).shuffle(kept)

    quality_metrics = _analyze_dataset_quality(kept)
    _save_jsonl(output_path, kept)
    print(f"✅ Unified {len(kept)} rows from {len(all_rows)} inputs into {output_path}")

    _print_quality_report(quality_metrics)


# --- Main Split Function ---

def split_dataset(
    input_path: pathlib.Path,
    train_path: pathlib.Path,
    eval_path: pathlib.Path,
    eval_pct: float,
    seed: int,
):
    """Splits a JSONL dataset into training and evaluation sets."""
    rows = list(_iter_jsonl([input_path]))
    if not rows:
        print(f"No rows found in {input_path} to split.")
        return

    random.Random(seed).shuffle(rows)
    
    n_eval = max(1, int(round(len(rows) * eval_pct)))
    n_eval = min(n_eval, len(rows) - 1) if len(rows) > 1 else 0

    train_set = rows[:-n_eval] if n_eval > 0 else rows
    eval_set = rows[-n_eval:] if n_eval > 0 else []

    _save_jsonl(train_path, train_set)
    _save_jsonl(eval_path, eval_set)
    print(f"✅ Split {len(rows)} rows into {len(train_set)} train and {len(eval_set)} eval files.")
