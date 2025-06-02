"""
FastAPI service for French NER anonymisation (WikiANN-FR).

All resources are local:
  • ./app/base_model  – merged CamemBERT backbone (7-label head)
"""

import sys, re, string
from pathlib import Path
from typing import List, Set, Dict, Optional
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

MASK_CATEGORIES = {
    "faces",          # laissé pour compatibilité (géré par le service image)
    "person_names",
    "organizations",
    "locations",
    "emails",
    "phones",
    "dates",
    "other",
}
# NER → catégories mask  ----------------------------------------------- 
NER_CATEGORY_MAP: Dict[str, Set[str]] = {
    "person_names":  {"PER"},
    "organizations": {"ORG"},
    "locations":     {"LOC"},
}

# Regex → catégories mask  --------------------------------------------- 
REGEX_CATEGORY_MAP: Dict[str, List[str]] = {
    "emails":  ["email"],
    "phones":  ["phone_fr", "phone_ch", "phone_int"],
    "dates":   ["date"],
    "other":   [
        "siret", "insee", "avs", "iban_fr", "iban_ch",
        "credit_card", "postal_code_fr", "postal_code_ch",
        "ip_v4", "mac", "url", "handle",
    ],
}

# ────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ────────────────────────────────────────────────────────────────────────────

from transformers.utils import logging
logging.set_verbosity_error()  # Réduire le bruit de transformers

tokenizer = None
model = None
ner_pipeline = None
print("[DEBUG] Contents of BASE_MODEL_PATH:", list(Path(BASE_MODEL_PATH).glob("*")))

if not Path(BASE_MODEL_PATH).is_dir():
    print("[DEBUG] Contents of BASE_MODEL_PATH:", list(Path(BASE_MODEL_PATH).glob("*")))

    print(f"[ERROR] Model path {BASE_MODEL_PATH} does not exist.", file=sys.stderr)
else:
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
        cfg = AutoConfig.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
        cfg.id2label, cfg.label2id = id2label, label2id

        model = AutoModelForTokenClassification.from_pretrained(
            BASE_MODEL_PATH, config=cfg, loc al_files_only=True
        )
        model.eval()

        ner_pipeline = pipeline(
            "ner", model=model, tokenizer=tokenizer, aggregation_strategy="average"
        )

        print("[INFO] NER model loaded successfully.")

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}", file=sys.stderr)
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

# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────


def _regex_entities(text: str, allowed: Set[str]) -> set[str]:          # MOD
    """Renvoie les hits Regex dont les clés sont dans *allowed*."""
    out = set()
    for lab, pat in REGEX_PATTERNS.items():
        if lab not in allowed:
            continue
        hits = re.findall(pat, text, flags=re.IGNORECASE | re.VERBOSE)
        if hits:
            print(f"[DEBUG] {lab}: {hits}", file=sys.stderr)
            out.update(map(str.strip, hits))
    return out


def _filter_entities(ents, raw: str, allowed_groups: Set[str]) -> set[str]:  # MOD
    spans = set()
    for e in ents:
        if e.get("entity_group") not in allowed_groups:
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
    mask: Optional[List[str]] = None  

@app.post("/compute")
async def anonymize_text(payload: TextInput):
    print("Textinput mask: ")
    print(payload.mask)
    # Catégories à masquer
    selected: Set[str] = (
        set(payload.mask or MASK_CATEGORIES) & MASK_CATEGORIES
    )
    if not selected:
        # rien sélectionné = on flag rien
        return {"anonymize": []}

    # Mapping NER selon masque
    allowed_ner_groups: Set[str] = {
        grp
        for cat, groups in NER_CATEGORY_MAP.items()
        if cat in selected
        for grp in groups
    }

    # Mapping regex selon masque 
    allowed_regex_keys: Set[str] = {
        key
        for cat, keys in REGEX_CATEGORY_MAP.items()
        if cat in selected
        for key in keys
    }

    # Inference NER + regex 
    ents         = ner_pipeline(payload.text)
    ner_items    = _filter_entities(ents, payload.text, allowed_ner_groups)
    regex_items  = _regex_entities(payload.text, allowed_regex_keys)

    # Nettoyage / union 
    
    to_redact = _clean(ner_items.union(regex_items))
    return {"anonymize": to_redact}


@app.exception_handler(HTTPException)
async def http_error(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
