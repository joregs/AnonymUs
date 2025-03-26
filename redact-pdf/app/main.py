from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
from app.redactor import modify_pdf

app = FastAPI()

@app.post("/compute")
async def redact_pdf(file: UploadFile = File(...), word: str = Form(...)):
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    output_path = modify_pdf(file_path, word)
    
    return {"redacted_file": output_path}