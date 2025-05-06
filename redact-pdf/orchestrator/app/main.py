
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import requests
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# App & templates
# ---------------------------------------------------------------------------
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ---------------------------------------------------------------------------
# Micro‑service endpoints (cluster‑local)
# ---------------------------------------------------------------------------
EXTRACT_URL = "http://extract-text-service.anonymus.svc.cluster.local/extract-text/compute"
ANONYMISATION_URL = "http://anonymisation-service.anonymus.svc.cluster.local/anonymisation/compute"
REDACT_URL = "http://redact-pdf-service.anonymus.svc.cluster.local/redact-pdf/compute"

TIMEOUT = 60  # seconds per micro‑service call


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/pipeline")
async def process_pipeline(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Chaîne complète : upload, extract, anonymise, redact, download."""
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(tempfile.gettempdir(), session_id)
    os.makedirs(session_dir, exist_ok=True)
    print(f"[{session_id}] 🗂️  Session directory : {session_dir}")

    # ---------------------------------------------------------------------
    # 0. Sauvegarde fichier d'entrée
    # ---------------------------------------------------------------------
    input_path = os.path.join(session_dir, file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"[{session_id}] 📄 File saved : {input_path}")

    # ---------------------------------------------------------------------
    # 1. Extraction de texte
    # ---------------------------------------------------------------------
    print(f"[{session_id}] 🔍 Calling extract‑text …")
    with open(input_path, "rb") as f:
        extract_resp = requests.post(
            EXTRACT_URL,
            files={"file": (file.filename, f, "application/pdf")},
            data={"session_id": session_id},
            timeout=TIMEOUT,
        )
    if not extract_resp.ok:
        raise HTTPException(status_code=502, detail="extract‑text service failed")

    extracted_text = extract_resp.text
    print(f"[{session_id}] 📝 Extracted first 200 chars\n{extracted_text[:200]}")

    # ---------------------------------------------------------------------
    # 2. Anonymisation
    # ---------------------------------------------------------------------
    print(f"[{session_id}] 🕵️  Calling anonymisation …")
    anonymise_resp = requests.post(
        ANONYMISATION_URL,
        json={"text": extracted_text},
        timeout=TIMEOUT,
    )
    if not anonymise_resp.ok:
        raise HTTPException(status_code=502, detail="anonymisation service failed")

    to_anonymize: list[str] = anonymise_resp.json().get("anonymize", [])
    print(f"[{session_id}] 🧼 Words to redact : {to_anonymize}")

    if not to_anonymize:
        print(f"[{session_id}] ⚠️  Nothing to redact → returning original file")
        background_tasks.add_task(shutil.rmtree, session_dir, ignore_errors=True)
        return FileResponse(input_path, filename=file.filename, media_type="application/pdf")

    # ---------------------------------------------------------------------
    # 3. Redaction (PDF)
    # ---------------------------------------------------------------------
    payload = {
        "session_id": session_id,
        "filename": file.filename,
        "word": to_anonymize,
    }
    print(f"[{session_id}] ✏️  Calling redact‑pdf …")
    redact_resp = requests.post(REDACT_URL, json=payload, timeout=TIMEOUT * 2)

    if not redact_resp.ok:
        print(
            f"[{session_id}] ❌ Redact service {redact_resp.status_code} → {redact_resp.text[:300]}"
        )
        raise HTTPException(status_code=502, detail="redact‑pdf service failed")

    # ---------------------------------------------------------------------
    # 4. Sauvegarde + réponse
    # ---------------------------------------------------------------------
    redacted_path = os.path.join(session_dir, f"redacted_{file.filename}")
    with open(redacted_path, "wb") as f:
        f.write(redact_resp.content)
    print(f"[{session_id}] ✅ Redacted PDF saved : {redacted_path}")

    # Nettoyage asynchrone après l’envoi
    background_tasks.add_task(shutil.rmtree, session_dir, ignore_errors=True)
    print(f"[{session_id}] 🧹 Cleanup scheduled")

    return FileResponse(
        redacted_path,
        filename=f"redacted_{file.filename}",
        media_type="application/pdf",
    )
