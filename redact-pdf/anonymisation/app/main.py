"""
FastAPI service for French NER anonymisation (WikiANN-FR).

All resources are local:
  • ./app/base_model  – merged CamemBERT backbone (7-label head)
"""

import sys, re, string
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, AutoConfig, pipeline
)
import nltk
from nltk.corpus import stopwords

# ────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────────────────────
BASE_MODEL_PATH  = "./app/base_model"
LABELS           = ["O","B-PER","I-PER","B-ORG","I-ORG","B-LOC","I-LOC"]
id2label         = {i: l for i, l in enumerate(LABELS)}
label2id         = {l: i for i, l in id2label.items()}
ENTITY_GROUPS    = {"PER", "ORG", "LOC"}
SCORE_THRESHOLD  = 0.01

# ────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ────────────────────────────────────────────────────────────────────────────
if not Path(BASE_MODEL_PATH).is_dir():
    raise RuntimeError("Model directory missing")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
cfg       = AutoConfig.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
cfg.id2label, cfg.label2id = id2label, label2id

model = AutoModelForTokenClassification.from_pretrained(
    BASE_MODEL_PATH, config=cfg, local_files_only=True
)
model.eval()

ner_pipeline = pipeline(
    "ner", model=model, tokenizer=tokenizer, aggregation_strategy="average"
)

# ────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ────────────────────────────────────────────────────────────────────────────
app = FastAPI(root_path="/anonymisation")

nltk.download("stopwords", quiet=True)
FRENCH_STOPWORDS = set(stopwords.words("french"))

# ────────────────────────────────────────────────────────────────────────────
# REGEX PATTERNS
# ────────────────────────────────────────────────────────────────────────────
REGEX_PATTERNS = {
    "email":         r"""\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b""",
    "phone_fr":      r"""\b(?:\+33|0)[1-9](?:[ .-]?\d{2}){4}\b""",
    "phone_ch":      r"""\b(?:\+41|0)(?:[ .-]?\d{2}){4}\b""",
    "phone_int":     r"""\b\+\d{1,3}(?:[ .-]?\d{1,4}){3,}\b""",
    "date":          r"""\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}\s?(?:janv\.?|févr\.?|mars|avr\.?|mai|juin|juil\.?|août|sept\.?|oct\.?|nov\.?|déc\.?|janvier|février|avril|juillet|septembre|octobre|novembre|décembre)(?:\s\d{2,4})?)\b""",
    "siret":         r"""\b\d{3}[ ]?\d{3}[ ]?\d{3}(?:\d{5})?\b""",
    "insee":         r"""\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{2}\b""",
    "avs":           r"""\b756\.\d{4}\.\d{4}\.\d{2}\b""",
    "iban_fr":       r"""\bFR\d{2}(?:\s?\d{4}){5}\b""",
    "iban_ch":       r"""\bCH\d{2}(?:\s?\d{5}){3,}\b""",
    "credit_card":   r"""\b(?:\d{4}[ -]?){3}\d{4}\b""",
    "postal_code_fr":r"""\b(?:0[1-9]|[1-8]\d|9[0-8])\d{3}\b""",
    "postal_code_ch":r"""\b\d{4}\b""",
    "ip_v4":         r"""\b(?:\d{1,3}\.){3}\d{1,3}\b""",
    "mac":           r"""\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b""",
    "url":           r"""https?://[^\s,\"\']+""",
    "handle":        r"""(?<!\w)@\w{3,}\b(?!\.\w)""",
}

def _regex_entities(text: str) -> set[str]:
    out = set()
    for lab, pat in REGEX_PATTERNS.items():
        hits = re.findall(pat, text, flags=re.IGNORECASE | re.VERBOSE)
        if hits:
            print(f"[DEBUG] {lab}: {hits}", file=sys.stderr)
            out.update(map(str.strip, hits))
    return out

# ────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────
def _filter_entities(ents, raw: str) -> set[str]:
    spans = set()
    for e in ents:
        if e.get("entity_group") not in ENTITY_GROUPS:
            continue
        if e.get("score", 0) < SCORE_THRESHOLD:
            continue
        start, end = e.get("start"), e.get("end")
        if not (isinstance(start, int) and isinstance(end, int)):
            continue
        text_span = raw[start:end].strip(" \n\r\t-–—")
        if (text_span and len(text_span) > 2
                and not all(c in string.punctuation for c in text_span)):
            spans.add(text_span)
    print(f"[DEBUG] NER spans: {sorted(spans)}", file=sys.stderr)
    return spans

def _clean(items):
    return sorted({
        t for t in items
        if (t and len(t) > 2 and
            t.lower() not in FRENCH_STOPWORDS and
            not all(c in string.punctuation for c in t))
    })

# ────────────────────────────────────────────────────────────────────────────
# API ROUTES
# ────────────────────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

@app.post("/compute")
async def anonymize_text(payload: TextInput):
    ents = ner_pipeline(payload.text)
    ner_items   = _filter_entities(ents, payload.text)
    regex_items = _regex_entities(payload.text)
    return {"anonymize": _clean(ner_items.union(regex_items))}

@app.exception_handler(HTTPException)
async def http_error(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
