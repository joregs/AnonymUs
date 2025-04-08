from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.responses import PlainTextResponse
import shutil, os
from app.redactor import modify_pdf
from app.redactor import extract_text

print(">>> FastAPI launched with root_path = /redact <<<")
app = FastAPI(root_path="/redact")

@app.post("/compute")
async def redact_pdf(file: UploadFile = File(...), word: str = Form(...)):
    input_path = f"/tmp/{file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = modify_pdf(input_path, word)

    return FileResponse(output_path, filename=os.path.basename(output_path), media_type='application/pdf')

@app.post("/PDFToText")
async def redact_pdf(file: UploadFile = File(...)):
    input_path = f"/tmp/{file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    word = extract_text(input_path)

    return PlainTextResponse(word)