# src/lora_sftuner/importers/twitter_api_importer.py

import datetime
import html
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests
from tqdm import tqdm

# Correctly import the shared processing functions
from ..processing import twitter_data

API_URL = "https://api.twitter.com/2"

# --- Helper Functions ---

def _load_state(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError):
            pass
    return {}

def _save_state(path: Path, state: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    state["last_run_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")

def _scan_existing_ids(out_path: Path) -> Set[str]:
    """Scans an existing JSONL file and returns a set of all tweet_ids."""
    if not out_path.exists():
        return set()
    ids = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(str(json.loads(line).get("tweet_id")))
            except (json.JSONDecodeError, AttributeError):
                continue
    return ids

def _get_user_id(username: str, bearer_token: str) -> str:
    """Fetches the Twitter user ID for a given username."""
    url = f"{API_URL}/users/by/username/{username}"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()["data"]["id"]

def _fetch_tweets_since(user_id: str, start_time: str, bearer_token: str, include_replies: bool):
    """Generator that yields tweet dicts from the Twitter API v2."""
    params = {
        "max_results": 100,
        "start_time": start_time,
        "tweet.fields": "id,text,created_at,source,referenced_tweets,lang",
        "exclude": "retweets" if include_replies else "retweets,replies",
    }
    headers = {"Authorization": f"Bearer {bearer_token}"}
    next_token = None
    
    with tqdm(desc="Fetching tweets", unit=" tweets") as pbar:
        while True:
            if next_token:
                params["pagination_token"] = next_token
            
            try:
                response = requests.get(f"{API_URL}/users/{user_id}/tweets", headers=headers, params=params, timeout=30)
                if response.status_code == 429: # Rate limit
                    reset = int(response.headers.get("x-rate-limit-reset", "0"))
                    wait = max(5, reset - int(time.time()))
                    print(f"Rate limited. Sleeping {wait}s...")
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                
                data = response.json()
                tweets = data.get("data", [])
                for t in tweets:
                    yield t
                pbar.update(len(tweets))
                
                next_token = data.get("meta", {}).get("next_token")
                if not next_token:
                    break
            except requests.RequestException as e:
                print(f"\nAPI request failed: {e}")
                break

# --- Main Processing Function ---

def sync_tweets(
    username: str,
    bearer_token: str,
    out_path: Path,
    state_path: Path,
    min_len: int,
    exclude_sources: Set[str],
    include_replies: bool,
    no_quotes: bool,
    role_assistant: str,
):
    """Main function to perform an incremental sync of new tweets."""
    if not bearer_token:
        print("Error: Twitter Bearer token is missing.")
        return

    state = _load_state(state_path)
    start_time = state.get("start_time", "1970-01-01T00:00:00Z")
    existing_ids = _scan_existing_ids(out_path)
    
    print(f"Syncing tweets for @{username} since {start_time}")
    user_id = _get_user_id(username, bearer_token)

    new_examples = []
    newest_time = start_time
    
    # Fetch all new tweets first
    all_new_tweets = list(_fetch_tweets_since(user_id, start_time, bearer_token, include_replies))
    
    # Now process them
    for tweet in tqdm(all_new_tweets, desc="Processing new tweets"):
        tweet_id = str(tweet.get("id", ""))
        if tweet_id in existing_ids:
            continue
        
        if no_quotes and any(ref.get("type") == "quoted" for ref in tweet.get("referenced_tweets", [])):
            continue

        if tweet.get("source") in exclude_sources:
            continue

        # Use the shared unifier and example builder
        unified_tweet = twitter_data.unify_tweet(tweet)
        if len(unified_tweet["text"]) < min_len:
            continue

        example = twitter_data.make_style_example(unified_tweet, "Write a tweet in my style.", role_assistant)
        if example:
            new_examples.append(example)
            existing_ids.add(tweet_id)
        
        created_at = tweet.get("created_at")
        if created_at and created_at > newest_time:
            newest_time = created_at

    if not new_examples:
        print("No new tweets found to append.")
        return

    print(f"Appending {len(new_examples)} new examples to {out_path.name}")
    with out_path.open("a", encoding="utf-8") as f:
        for ex in new_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    try:
        dt = datetime.datetime.fromisoformat(newest_time.replace("Z", "+00:00"))
        next_start_time = (dt + datetime.timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    except ValueError:
        next_start_time = newest_time

    state["start_time"] = next_start_time
    _save_state(state_path, state)
    print(f"✅ Sync complete. Next sync will start from {next_start_time}")

