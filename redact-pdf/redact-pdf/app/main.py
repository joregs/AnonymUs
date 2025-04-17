print(">>> ✅ FastAPI launched with root_path = /redact-pdf")

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, PlainTextResponse
from datetime import datetime
import shutil, os, uuid, traceback
from app.redactor import modify_pdf

app = FastAPI(root_path="/redact-pdf")

def log(msg):
    print(f"[{datetime.now().isoformat()}] {msg}")

@app.post("/compute")
async def redact_pdf(file: UploadFile = File(...), word: str = Form(...)):
    session_id = str(uuid.uuid4())
    log(f"[{session_id}] 🚀 New /compute request received")
    log(f"[{session_id}] 🗂️ Filename: {file.filename}")
    log(f"[{session_id}] 🔒 Redaction targets (raw): {word}")

    try:
        input_path = f"/tmp/{file.filename}"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        log(f"[{session_id}] 📄 File saved to: {input_path}")

        output_path = modify_pdf(input_path, word)
        log(f"[{session_id}] ✅ Redacted PDF path: {output_path}")

        return FileResponse(
            output_path,
            filename=os.path.basename(output_path),
            media_type='application/pdf'
        )

    except Exception as e:
        log(f"[{session_id}] ❌ Exception occurred during redaction")
        log(traceback.format_exc())
        return PlainTextResponse(f"Internal error: {str(e)}", status_code=500)
