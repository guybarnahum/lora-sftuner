# src/lora_sftuner/importers/docs_importer.py

import datetime
import hashlib
import html
import json
import pathlib
import re
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

# --- Optional Dependency Handling ---
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import docx  # python-docx
except ImportError:
    docx = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

SUPPORTED_EXT = {".txt", ".md", ".markdown", ".html", ".htm", ".docx", ".pdf"}

# --- Text Processing and Chunking ---

def _norm_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _norm_text_for_hash(t: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(t or "")).strip().lower()

def _sha256_text(t: str) -> str:
    return hashlib.sha256(_norm_text_for_hash(t).encode("utf-8")).hexdigest()

def _topic_keywords(t: str, limit=8) -> List[str]:
    t = re.sub(r"https?://\S+", "", t)
    words = re.findall(r"[A-Za-z\u0590-\u05FF][\w’׳״'-]{2,}", t)
    out = {w.lower() for w in words}
    return list(out)[:limit] or ["general"]

def _split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in parts if p]

# --- Document Readers ---

def _read_txt(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp1255", errors="ignore")

def _read_html(path: pathlib.Path) -> str:
    if not BeautifulSoup:
        raise ImportError("HTML processing requires 'beautifulsoup4'. Please install with `pip install -e \".[docs]\"`")
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n")

def _read_docx(path: pathlib.Path) -> str:
    if not docx:
        raise ImportError("DOCX processing requires 'python-docx'. Please install with `pip install -e \".[docs]\"`")
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def _read_pdf(path: pathlib.Path) -> str:
    if not fitz:
        raise ImportError("PDF processing requires 'PyMuPDF'. Please install with `pip install -e \".[docs]\"`")
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text

def _load_text_from_file(path: pathlib.Path) -> str:
    """Dispatcher to call the correct reader based on file extension."""
    ext = path.suffix.lower()
    if ext == ".txt": return _read_txt(path)
    if ext in {".md", ".markdown"}: return _read_txt(path) # Treat as plain text, user can strip markdown later if needed
    if ext in {".html", ".htm"}: return _read_html(path)
    if ext == ".docx": return _read_docx(path)
    if ext == ".pdf": return _read_pdf(path)
    return ""

# --- SFT Example Builders ---

def _make_style_example(txt: str, src_path: str, lang: str) -> Dict:
    topics = _topic_keywords(txt)
    prompt = f"Write a paragraph in my signature style about: {', '.join(topics)}."
    return {
        "source": src_path,
        "lang": lang,
        "messages": [{"role": "user", "content": prompt}, {"role": "model", "content": txt}],
    }

# --- State Management ---

def _load_state(path: pathlib.Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError):
            pass
    return {"files": {}}

def _save_state(path: pathlib.Path, state: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    state["last_run_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")

# --- Main Processing Function ---

def process_documents(
    root_path: pathlib.Path,
    out_path: pathlib.Path,
    state_path: pathlib.Path,
    min_chars: int,
    max_chars: int,
    tag_lang: str,
    delete_missing: bool,
):
    """Incrementally ingests documents from a directory into SFT JSONL format."""
    state = _load_state(state_path)
    
    files_to_process = [
        p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    ]
    
    added_count = 0
    seen_files = set()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as out_f:
        for fp in tqdm(files_to_process, desc="Processing documents"):
            file_key = str(fp.resolve())
            seen_files.add(file_key)
            stat = fp.stat()
            
            info = state["files"].get(file_key, {})
            if info.get("mtime") == stat.st_mtime and info.get("size") == stat.st_size:
                continue # Skip unchanged files

            try:
                text = _load_text_from_file(fp)
                if not text: continue
            except Exception as e:
                print(f"\n[Warning] Failed to read {fp.name}: {e}")
                continue

            new_hashes = set()
            paragraphs = _split_paragraphs(_norm_ws(text))
            
            for par in paragraphs:
                if len(par) < min_chars or len(par) > max_chars:
                    continue # Simple chunking, can be improved later
                
                h = _sha256_text(par)
                if h in info.get("chunks", []):
                    new_hashes.add(h)
                    continue

                example = _make_style_example(par, str(fp), tag_lang)
                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                added_count += 1
                new_hashes.add(h)

            state["files"][file_key] = {"mtime": stat.st_mtime, "size": stat.st_size, "chunks": sorted(list(new_hashes))}

    if delete_missing:
        keys_to_prune = set(state["files"].keys()) - seen_files
        for key in keys_to_prune:
            del state["files"][key]
            print(f"Pruned missing file from state: {key}")

    _save_state(state_path, state)
    print(f"✅ Appended {added_count} new document chunks to {out_path}")


