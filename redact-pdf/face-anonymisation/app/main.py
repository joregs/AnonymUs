print(">>> ✅ FastAPI launched with root_path = /blur-faces")

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, PlainTextResponse
import shutil, uuid
from app.blur_faces import blur_faces

app = FastAPI(root_path="/blur-faces")

@app.post("/compute")
async def process_pdf(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    input_path = f"/tmp/input-{session_id}.pdf"
    output_path = f"/tmp/output-{session_id}.pdf"
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    blur_faces(input_path, output_path)
    
    return FileResponse(
        output_path,
        filename=f"blurred-{file.filename}",
        media_type='application/pdf'
    )
