# src/lora_sftuner/importers/twitter_importer.py

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

from tqdm import tqdm

# Correctly import the shared processing functions from the module in the Canvas
from ..processing import twitter_data

def process_archive(
    archive_path: Path,
    out_path: Path,
    eval_pct: float,
    include_replies: bool,
    no_quotes: bool,
    dialog: bool,
    role_assistant: str,
):
    """Processes a full Twitter archive into SFT JSONL format."""
    print(f"Loading archive from: {archive_path}")
    raw_tweets = twitter_data.load_archive(archive_path)
    
    unified = [twitter_data.unify_tweet(t) for t in tqdm(raw_tweets, desc="Normalizing tweets")]
    by_id = {t["id_str"]: t for t in unified if t["id_str"]}
    
    kept, seen_hashes = [], set()
    for tw in tqdm(unified, desc="Filtering & Deduplicating"):
        if tw["is_retweet"]: continue
        if no_quotes and tw["is_quote"]: continue
        if not include_replies and tw["in_reply_to_id"]: continue
        
        h = hashlib.sha256(twitter_data.norm_for_dedup(tw["text"]).encode()).hexdigest()
        if h in seen_hashes: continue
        seen_hashes.add(h)
        kept.append(tw)
        
    kept.sort(key=lambda x: x["ts"] or datetime(1970, 1, 1, tzinfo=timezone.utc))
    
    rows = []
    for tw in tqdm(kept, desc="Building SFT examples"):
        example = None
        if dialog and tw["in_reply_to_id"]:
            example = twitter_data.make_dialog_example(tw, by_id, 1, role_assistant)
        
        if not example:
            example = twitter_data.make_style_example(tw, "Write a tweet in my style.", role_assistant)
            
        rows.append(example)

    if not rows:
        print("No tweets to write after filtering.")
        return

    n_eval = int(round(len(rows) * eval_pct))
    train_rows, eval_rows = (rows[:-n_eval], rows[-n_eval:]) if n_eval > 0 else (rows, [])
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ Wrote {len(train_rows)} training examples to {out_path}")

    if eval_rows:
        eval_path = out_path.with_name(f"{out_path.stem}_eval.jsonl")
        with eval_path.open("w", encoding="utf-8") as f:
            for r in eval_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ Wrote {len(eval_rows)} evaluation examples to {eval_path}")

