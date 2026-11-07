"""Public E1 plus a relative Gemma item rating that checks itself against labels.

    s    = (5 - rating) * SCALE
    x    = clip(m / (m + 3) * (s - mean_s), -CAP, CAP)
    beta = clip((LAM * BETA0 + g) / (LAM + H), 0, BETA_MAX)
    p    = sigmoid(logit(p_E1) + beta * x)

rating is the expected digit, 1 (trivial) to 9 (extremely hard), from one forward
pass of google/gemma-4-31B-it on the visible item. mean_s and m come from rated
labeled items on this benchmark. g and H are the score and information of a logistic
slope at zero on those labels. With no labels x is 0, so budget zero equals E1 and
the model is never loaded. Ratings are cached by item hash and hold no labels. If
the GPU, the model or a rating fails, that item gets no shift, which leaves E1.
"""
import hashlib
import json
import math
import os
import tempfile
import threading
import time
from pathlib import Path

import e1_base

MODEL = "google/gemma-4-31B-it"
SCALE = 1.2 / 2.218
BETA0 = 0.6
LAM = 11.0
CAP = 1.5
BETA_MAX = 1.2
HEAD, TAIL = 1000, 2000
BATCH = 4
LOAD_WAIT = 60
RATE_WINDOW = 30
TASK = "Below is a task from an AI evaluation benchmark.\n\n{meta}<task>\n{t}\n</task>\n\n"
ASK = ("How hard is this task for a strong AI system (a frontier LLM or LLM agent) to solve correctly? "
       "Answer with a single digit from 1 (trivial, almost every system solves it) to 9 "
       "(extremely hard, almost no system solves it).")

_memory = {}
_engine = {}
_ready = threading.Event()
_load_lock = threading.Lock()
_started = False


def _writable(path):
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".probe.{os.getpid()}"
        probe.write_text("1")
        probe.unlink()
        return True
    except OSError:
        return False


def _cache_dir():
    configured = os.environ.get("PAIEC_V4_CACHE")
    candidates = [Path(configured)] if configured else []
    candidates += [Path(__file__).with_name(".v4_ratings"), Path("/tmp/v4_ratings"), Path(tempfile.gettempdir()) / "v4_ratings"]
    for path in candidates:
        if _writable(path):
            return path
    return None


_cache = _cache_dir()


def _load():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if torch.cuda.is_available():
            tok = AutoTokenizer.from_pretrained(MODEL)
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            net = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
            digits = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(1, 10)]
            _engine.update(torch=torch, tok=tok, net=net, digits=digits)
    except Exception:
        _engine.clear()
    _ready.set()


def _start_loading():
    global _started
    with _load_lock:
        if not _started:
            _started = True
            threading.Thread(target=_load, daemon=True).start()


def _prompt(tok, item):
    content, features = item["item_content"], item.get("item_features") or ""
    ids = tok.encode(content, add_special_tokens=False)
    text = content if len(ids) <= HEAD + TAIL else tok.decode(ids[:HEAD]) + "\n[...]\n" + tok.decode(ids[-TAIL:])
    meta = f"Metadata: {features}\n\n" if features else ""
    message = [{"role": "user", "content": TASK.format(meta=meta, t=text) + ASK}]
    return tok.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _rate_batch(items):
    """Return one expected digit per item. Replaced by a stub in tests."""
    torch, tok, net, digits = _engine["torch"], _engine["tok"], _engine["net"], _engine["digits"]
    encoded = [tok(_prompt(tok, item), add_special_tokens=False).input_ids for item in items]
    batch = tok.pad({"input_ids": encoded}, return_tensors="pt").to(net.device)
    positions = (batch.attention_mask.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        logits = net(**batch, position_ids=positions, use_cache=False, logits_to_keep=1).logits[:, -1]
        probs = torch.softmax(logits.float(), -1)[:, digits]
        probs = probs / probs.sum(1, keepdim=True)
        ratings = (probs * torch.arange(1, 10, device=probs.device, dtype=probs.dtype)).sum(1)
    return ratings.cpu().tolist()


def _key(item):
    return hashlib.sha256(f"{item['item_content']}\0{item.get('item_features') or ''}".encode()).hexdigest()


def _stored(key):
    if key in _memory:
        return _memory[key]
    if _cache is not None:
        try:
            value = json.loads((_cache / (key + ".json")).read_text())["rating"]
            _memory[key] = value
            return value
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return None


def _save(key, rating):
    _memory[key] = rating
    if _cache is not None:
        temp = _cache / f"{key}.{os.getpid()}.tmp"
        try:
            temp.write_text(json.dumps(dict(rating=rating)))
            os.replace(temp, _cache / (key + ".json"))
        except OSError:
            pass


def _ensure_rated(items):
    missing = {}
    for item in items:
        key = _key(item)
        if key not in missing and _stored(key) is None:
            missing[key] = item
    if not missing:
        return
    if not _engine:
        _start_loading()
        _ready.wait(LOAD_WAIT)
    if not _engine and _ready.is_set():
        return
    if not _engine:
        return
    keys = list(missing)
    deadline = time.monotonic() + RATE_WINDOW
    for start in range(0, len(keys), BATCH):
        if time.monotonic() > deadline:
            break  # The target comes first, and later calls rate the rest.
        chunk = keys[start:start + BATCH]
        try:
            ratings = _rate_batch([missing[k] for k in chunk])
        except Exception:
            try:
                _engine["torch"].cuda.empty_cache()
            except Exception:
                pass
            continue
        for key, rating in zip(chunk, ratings):
            if math.isfinite(rating):
                _save(key, min(max(float(rating), 1.0), 9.0))


def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _score(item):
    rating = _stored(_key(item))
    return None if rating is None else (5 - rating) * SCALE


def predict(input: list[dict], labeled: list | None = None) -> float:
    subject, item = input
    base = e1_base.predict(input, labeled)
    benchmark = item.get("benchmark_id") if isinstance(item, dict) else None
    if not isinstance(benchmark, str) or not benchmark:
        return base

    rows = []
    for entry in labeled or ():
        try:
            (other, row), y = entry
        except (TypeError, ValueError):
            continue
        if isinstance(row, dict) and row.get("benchmark_id") == benchmark and y in (0, 1, 0.0, 1.0):
            rows.append((json.dumps(other, sort_keys=True, default=str), row, int(y)))
    if not rows:
        return base

    unique = {_key(r): r for _, r, _ in rows}
    _ensure_rated([item, *unique.values()])
    s = _score(item)
    known = {k: _score(r) for k, r in unique.items()}
    known = {k: v for k, v in known.items() if v is not None}
    if s is None or not known:
        return base
    m = len(known)
    mean = math.fsum(known.values()) / m

    totals = {}
    for name, _, y in rows:
        k, count = totals.get(name, (0, 0))
        totals[name] = (k + y, count + 1)
    g = h = 0.0
    for name, row, y in rows:
        value = known.get(_key(row))
        if value is None:
            continue
        k, count = totals[name]
        rate = (k + 1.5) / (count + 3)
        x = min(max(value - mean, -CAP), CAP)
        g += (y - rate) * x
        h += rate * (1 - rate) * x * x
    beta = min(max((LAM * BETA0 + g) / (LAM + h), 0.0), BETA_MAX)
    x = min(max(m / (m + 3) * (s - mean), -CAP), CAP)
    z = _logit(base) + beta * x
    return float(min(max(1 / (1 + math.exp(-z)), 1e-6), 1 - 1e-6))
