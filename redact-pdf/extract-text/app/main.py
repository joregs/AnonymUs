print(">>> FastAPI launched with root_path = /extract-text")

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
import shutil, os, uuid
from app.extractor import extract_text
import boto3
from botocore.client import Config

# Configuration MinIO
endpoint_url = "https://minio-anonymus.kube-ext.isc.heia-fr.ch"
access_key = "admin"
secret_key = "SuperAnonym"
bucket_name = "pdfs"

app = FastAPI(root_path="/extract-text")

def upload_pdf_to_s3(file_path, session_id, filename):
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4'),
        region_name="us-east-1"
    )
    try:
        s3.head_bucket(Bucket=bucket_name)
    except:
        s3.create_bucket(Bucket=bucket_name)

    object_name = f"sessions/{session_id}/{filename}"
    s3.upload_file(file_path, bucket_name, object_name)
    print(f"File '{file_path}' uploaded to '{bucket_name}/{object_name}'")
    return object_name

@app.post("/compute")
async def extract_and_upload(session_id: str = Form(...), file: UploadFile = File(...)):
    # session_id = str(uuid.uuid4())
    print(f"[{session_id}] Starting new session")

    input_path = f"/tmp/{session_id}_{file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"[{session_id}] File saved to {input_path}")

    object_key = upload_pdf_to_s3(input_path, session_id, file.filename)

    extracted_text = extract_text(input_path)
    print(f"[{session_id}] Text extracted")

    return PlainTextResponse(extracted_text)
