from fastapi import FastAPI, UploadFile, File, Request, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil, os, requests, tempfile, uuid
from pathlib import Path

app = FastAPI()

# Template and static setup
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# URLs for microservices
EXTRACT_URL = "http://extract-text-service.anonymus.svc.cluster.local/extract-text/compute"
ANONYMISATION_URL = "http://anonymisation-service.anonymus.svc.cluster.local/anonymisation/compute"
REDACT_URL = "http://redact-pdf-service.anonymus.svc.cluster.local/redact-pdf/compute"

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/pipeline")
async def process_pipeline(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(tempfile.gettempdir(), session_id)
    os.makedirs(session_dir, exist_ok=True)
    print(f"[{session_id}] 🗂️ Session directory created at: {session_dir}")

    input_path = os.path.join(session_dir, file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"[{session_id}] 📄 Uploaded file saved to: {input_path}")

    # Step 1: Extract text
    print(f"[{session_id}] 🔍 Sending file to extract-text...")
    with open(input_path, "rb") as f:
        extract_response = requests.post(EXTRACT_URL, files={"file": (file.filename, f)})
    print(f"[{session_id}] 📝 Extract status: {extract_response.status_code}")
    extracted_text = extract_response.text
    print(f"[{session_id}] 🧾 Extracted text (first 300 chars):\n{extracted_text[:300]}")

    # Step 2: Anonymize
    print(f"[{session_id}] 🕵️ Sending text to anonymisation...")
    anonymise_response = requests.post(
        ANONYMISATION_URL,
        json={"text": extracted_text}
    )
    print(f"[{session_id}] 🔐 Anonymisation status: {anonymise_response.status_code}")
    print(f"[{session_id}] 🔍 Raw anonymisation response: {anonymise_response.text}")

    to_anonymize = anonymise_response.json().get("anonymize", [])
    print(f"[{session_id}] 🧼 Words to redact: {to_anonymize}")

    # If nothing to redact
    if not to_anonymize:
        print(f"[{session_id}] ⚠️ No words found to redact. Returning original file.")
        background_tasks.add_task(shutil.rmtree, session_dir, ignore_errors=True)
        return FileResponse(input_path, filename=file.filename, media_type='application/pdf')

    # Step 3: Redact
    redact_word_list = ",".join(to_anonymize)
    print(f"[{session_id}] 🖊️ Sending words to redact-pdf: {redact_word_list}")
    with open(input_path, "rb") as f:
        redact_response = requests.post(
            REDACT_URL,
            files={"file": (file.filename, f)},
            data={"word": redact_word_list}
        )
    print(f"[{session_id}] 🧾 Redact status: {redact_response.status_code}")

    redacted_path = os.path.join(session_dir, f"redacted_{file.filename}")
    with open(redacted_path, "wb") as out_f:
        out_f.write(redact_response.content)
    print(f"[{session_id}] ✅ Redacted PDF saved to: {redacted_path}")

    # Cleanup after response
    background_tasks.add_task(shutil.rmtree, session_dir, ignore_errors=True)
    print(f"[{session_id}] 🧹 Cleanup scheduled")

    return FileResponse(redacted_path, filename=os.path.basename(redacted_path), media_type='application/pdf')
