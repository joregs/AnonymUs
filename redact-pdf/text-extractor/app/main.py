print(">>> ✅ FastAPI launched with root_path = /extract-text")

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.responses import PlainTextResponse
import shutil, os
# from app.extractor import modify_pdf
from app.extractor import extract_text

app = FastAPI(root_path="/extract-text")

# @app.post("/compute")
# async def redact_pdf(file: UploadFile = File(...), word: str = Form(...)):
#     input_path = f"/tmp/{file.filename}"
#     with open(input_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     output_path = modify_pdf(input_path, word)

#     return FileResponse(output_path, filename=os.path.basename(output_path), media_type='application/pdf')

@app.post("/compute")
async def redact_pdf(file: UploadFile = File(...)):
    input_path = f"/tmp/{file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    word = extract_text(input_path)

    return PlainTextResponse(word)