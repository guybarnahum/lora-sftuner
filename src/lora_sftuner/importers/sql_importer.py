# src/lora_sftuner/importers/sql_importer.py

import sqlite3
import re
import html
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from tqdm import tqdm

# --- Helper Functions ---

def _html_to_text(s: str) -> str:
    if not s: return ""
    s = s.replace("<br />", "\n").replace("<br/>", "\n").replace("<br>", "\n")
    s = re.sub(r"</p>\s*<p>", "\n\n", s, flags=re.I)
    s = re.sub(r"</?p[^>]*>", "", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("\\/", "/")
    s = html.unescape(s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def _connect_input(path: Path) -> sqlite3.Connection:
    """Connects to a SQLite DB file or an in-memory DB from a .sql dump."""
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        con = sqlite3.connect(":memory:")
        sql = path.read_text(encoding="utf-8", errors="ignore")
        con.executescript(sql)
    con.row_factory = sqlite3.Row
    return con

def _ancestors(by_id: Dict[Any, sqlite3.Row], row: sqlite3.Row, col_map: Dict[str, str]) -> List[sqlite3.Row]:
    """Returns a list [root, ..., this_row] by following the parent_id chain."""
    chain, seen = [], set()
    cur = row
    id_col, parent_id_col = col_map['id'], col_map['parent_id']
    
    while cur and cur[id_col] not in seen:
        chain.append(cur)
        seen.add(cur[id_col])
        pid = cur[parent_id_col]
        if not pid or pid == 0 or pid == cur[id_col]:
            break
        cur = by_id.get(pid)
    chain.reverse()
    return chain

# --- Main Processing Function ---

def process_database(
    db_path: Path,
    out_path: Path,
    user_nick: str,
    max_context: int,
    strip_self_context: bool,
    role_assistant: str,
):
    """
    Processes a SQL database to generate SFT examples by automatically
    loading a sidecar .yaml configuration file.
    """
    yaml_path = db_path.with_suffix(".yaml")
    if not yaml_path.exists():
        print(f"Error: Configuration file not found for '{db_path.name}'. Expected: '{yaml_path.name}'")
        return

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    col_map = config.get("schema_mapping", {}).get("column_names", {})
    table_name = config.get("schema_mapping", {}).get("table_name")

    if not all([col_map, table_name, user_nick]):
        print(f"Error: SQL importer configuration in '{yaml_path.name}' is incomplete.")
        return

    con = _connect_input(db_path)
    query = f"SELECT * FROM {table_name} ORDER BY datetime({col_map['created_at']}) ASC, {col_map['id']} ASC"
    
    print(f"Executing query: {query}")
    rows = list(con.execute(query))
    by_id = {r[col_map['id']]: r for r in rows}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in tqdm(rows, desc=f"Processing {db_path.name}"):
            if (r[col_map['author_nick']] or "") != user_nick:
                continue

            chain = _ancestors(by_id, r, col_map)
            if not chain: continue

            ctx = chain[:-1]
            if strip_self_context:
                ctx = [x for x in ctx if (x[col_map['author_nick']] or "") != user_nick]
            if max_context > 0:
                ctx = ctx[-max_context:]

            msgs = []
            for m in ctx:
                # *** FIX: Use dictionary-style access and check for key existence ***
                title_col = col_map.get('content_title')
                title = m[title_col] if title_col and title_col in m.keys() else ""
                
                body_col = col_map.get('content_body')
                body = m[body_col] if body_col and body_col in m.keys() else ""
                
                content = (title + "\n\n" + body).strip()
                if content:
                    msgs.append({"role": "user", "content": content})

            final = chain[-1]
            # *** FIX: Use dictionary-style access for the final message as well ***
            title_col = col_map.get('content_title')
            title = final[title_col] if title_col and title_col in final.keys() else ""
            
            body_col = col_map.get('content_body')
            body = final[body_col] if body_col and body_col in final.keys() else ""

            content = (title + "\n\n" + body).strip()
            if not content: continue
            msgs.append({"role": role_assistant, "content": content})

            if len(msgs) < 2: continue

            ex = {
                "thread_id": final[col_map['root_id']],
                "post_id": final[col_map['id']],
                "created_at": final[col_map['created_at']],
                "messages": msgs,
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n_written += 1
            
    print(f"✅ Wrote {n_written} samples from {db_path.name} to {out_path}")

