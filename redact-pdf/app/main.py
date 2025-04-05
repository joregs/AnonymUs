from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import shutil, os
from app.redactor import modify_pdf

print(">>> FastAPI launched with root_path = /redact <<<")
app = FastAPI(root_path="/redact")

@app.post("/compute")
async def redact_pdf(file: UploadFile = File(...), word: str = Form(...)):
    input_path = f"/tmp/{file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = modify_pdf(input_path, word)

    return FileResponse(output_path, filename=os.path.basename(output_path), media_type='application/pdf')
