import os
import traceback
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, validator
from typing import List, Union
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from app.blur_faces import blur_faces

app = FastAPI(root_path="/face-anonymisation")

# ------------------------------------------------------------------ #
# S3 / MinIO config
# ------------------------------------------------------------------ #
endpoint_url = "https://minio-anonymus.kube-ext.isc.heia-fr.ch"
access_key   = "admin"
secret_key   = "SuperAnonym"
bucket_name  = "pdfs"

def log(msg: str):
    print(f"[{datetime.now().isoformat()}] {msg}")

# ------------------------------------------------------------------ #
# S3 helpers (tout en str)
# ------------------------------------------------------------------ #
def download_pdf_from_s3(local_path: str, session_id: str, filename: str) -> bool:
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        region_name="us-east-1",
    )
    object_key = f"sessions/{session_id}/{filename}"
    try:
        s3.download_file(bucket_name, object_key, local_path)
        log(f"File '{object_key}' downloaded to '{local_path}'")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            log(f"File '{object_key}' not found in bucket '{bucket_name}'")
            return False
        raise

# ------------------------------------------------------------------ #
# Payload model
# ------------------------------------------------------------------ #
class RedactPayload(BaseModel):
    session_id: str
    filename:  str
    mask_faces: bool

# ------------------------------------------------------------------ #
# Endpoint
# ------------------------------------------------------------------ #
@app.post("/compute")
async def process_pdf(payload: RedactPayload):
    session_id = payload.session_id
    filename   = payload.filename         
    log(f"📥 request: session_id={session_id}, filename={filename}")

    input_path = f"/tmp/{filename}"

    if not download_pdf_from_s3(input_path, session_id, filename):
        return PlainTextResponse("File not found in S3", status_code=404)

    out_path = f"/tmp/blurred_{filename}"

    if payload.mask_faces : 
        try:
            blur_faces(input_path, out_path)
        except Exception:
            log(traceback.format_exc())
            raise HTTPException(500, "Face blurring failed")

        log(f"✅ faces blurred: {out_path}")

        return FileResponse(
            path=out_path,
            filename=os.path.basename(out_path),
            media_type="application/pdf",
        )
    else :
        return FileResponse(
            path=input_path,
            filename=os.path.basename(input_path),
            media_type="application/pdf",
        )
