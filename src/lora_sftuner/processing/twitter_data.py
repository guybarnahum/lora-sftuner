# src/lora_sftuner/processing/twitter_data.py

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Text Cleaning and Normalization ---

URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]+>")
A_TEXT_RE = re.compile(r">([^<]+)<")

TW_TIME_FMTS = (
    "%a %b %d %H:%M:%S %z %Y",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
)

def clean_text(t: str) -> str:
    if not t: return ""
    t = html.unescape(t)
    t = URL_RE.sub("", t)
    t = re.sub(r"[#@]\w+", "", t)
    t = re.sub(r'this tweet was edited at\s.*$', '', t, flags=re.IGNORECASE)
    t = t.replace("\u200f", "").replace("\u200e", "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = WS_RE.sub(" ", t).strip()
    return t

def norm_for_dedup(t: str) -> str:
    return clean_text(t).lower()

def parse_source_app(src_html: Optional[str]) -> str:
    if not src_html: return ""
    m = A_TEXT_RE.search(src_html)
    return clean_text(m.group(1)) if m else clean_text(TAG_RE.sub("", src_html))

def parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s: return None
    s = s.strip().replace("Z", "+00:00")
    for fmt in TW_TIME_FMTS:
        try:
            dt = datetime.strptime(s, fmt)
            if not dt.tzinfo: dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
    try:
        dt = datetime.fromisoformat(s)
        if not dt.tzinfo: dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

# --- Data Loading ---

def _load_js(path: Path) -> List[Dict[str, Any]]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    i = txt.find("[")
    j = txt.rfind("]")
    payload = txt[i : j + 1] if (i != -1 and j != -1 and j > i) else txt
    data = json.loads(payload)
    return [it.get("tweet", it) for it in data]

def load_archive(path: Path) -> List[Dict[str, Any]]:
    """Loads tweets from an archive folder or a single file."""
    if path.is_file():
        if path.suffix.lower() == ".js": return _load_js(path)
        raise ValueError(f"Unsupported file type for archive: {path}")

    candidates = list((path / "data").glob("tweets.js")) + list((path / "data" / "tweets").glob("*.js"))
    if not candidates:
        raise FileNotFoundError(f"No tweets.js found under: {path}")
    
    tweets = []
    for p in sorted(candidates):
        tweets.extend(_load_js(p))
    return tweets

# --- Tweet Processing ---

def unify_tweet(t: Dict[str, Any]) -> Dict[str, Any]:
    """Unifies different tweet formats into a single, consistent dictionary structure."""
    text = clean_text(t.get("full_text") or t.get("text") or "")
    return {
        "id_str": t.get("id_str") or str(t.get("id", "")),
        "text": text,
        "created_at": t.get("created_at"),
        "ts": parse_ts(t.get("created_at")),
        "lang": t.get("lang", ""),
        "source_app": parse_source_app(t.get("source")),
        "in_reply_to_id": t.get("in_reply_to_status_id_str") or str(t.get("in_reply_to_status_id") or ""),
        "is_quote": bool(t.get("is_quote_status") or t.get("quoted_status_id_str")),
        "is_retweet": text.startswith("RT @"),
    }

def make_style_example(tweet: Dict[str, Any], prompt: str, role_assistant: str) -> Dict[str, Any]:
    """Creates a simple 'style' SFT example."""
    return {
        "tweet_id": tweet["id_str"],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": role_assistant, "content": tweet["text"]},
        ],
    }

def make_dialog_example(tweet: Dict[str, Any], by_id: Dict[str, Dict[str, Any]], max_context: int, role_assistant: str) -> Optional[Dict[str, Any]]:
    """Creates a conversational SFT example by walking up the reply chain."""
    chain = [tweet]
    current_tweet = tweet
    for _ in range(max_context):
        parent_id = current_tweet.get("in_reply_to_id")
        if not parent_id or parent_id not in by_id:
            break
        parent_tweet = by_id[parent_id]
        chain.append(parent_tweet)
        current_tweet = parent_tweet
    
    chain.reverse()
    
    messages = []
    for i, item in enumerate(chain):
        role = "user" if (len(chain) - 1 - i) % 2 == 1 else role_assistant
        messages.append({"role": role, "content": item["text"]})

    if not messages or messages[-1]["role"] != role_assistant:
        return None

    return {
        "tweet_id": tweet["id_str"],
        "messages": messages,
    }

