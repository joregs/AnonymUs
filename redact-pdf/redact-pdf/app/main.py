print(">>> ✅ FastAPI launched with root_path = /redact-pdf")

from datetime import datetime
from typing import List, Union
import os
import traceback

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, validator

from app.redactor import modify_pdf 


endpoint_url = "https://minio-anonymus.kube-ext.isc.heia-fr.ch"
access_key = "admin"
secret_key = "SuperAnonym"
bucket_name = "pdfs"

app = FastAPI(root_path="/redact-pdf")

def log(msg):
    print(f"[{datetime.now().isoformat()}] {msg}")




def download_pdf_from_s3(local_path, session_id, filename):
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        region_name="us-east-1",
    )
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"sessions/{session_id}/")
    print([o["Key"] for o in objects.get("Contents", [])])

    object_name = f"sessions/{session_id}/{filename}"
    try:
        s3.download_file(bucket_name, object_name, local_path)
        print(f"File '{object_name}' downloaded to '{local_path}'")
        return local_path
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            print(f"File '{object_name}' not found in bucket '{bucket_name}'")
            return None          # triggers your 404 response
        raise                   # anything else really *is* an error

class RedactPayload(BaseModel):
    session_id: str
    filename: str
    word: Union[str, List[str]]  # list ou chaîne

    @validator("word", pre=True)
    def _ensure_list(cls, v):  # noqa: N805
        if isinstance(v, list):
            return v
        # autorise virgules ou points‑virgules comme séparateurs
        return [w.strip() for w in str(v).replace(";", ",").split(",") if w.strip()]



@app.post("/compute", response_class=FileResponse)
async def redact_pdf(payload: RedactPayload):
    session_id = payload.session_id
    filename = payload.filename
    words = payload.word  # toujours une liste grâce au validateur

    log(f"📥 Requête reçue : session_id={session_id}, filename={filename}, words={words}")

    try:
        input_path = f"/tmp/{filename}"

        if download_pdf_from_s3(input_path, session_id, filename) is None:
            return PlainTextResponse("File not found in S3", status_code=404)

        output_path = modify_pdf(input_path, words)
        log(f"✅ PDF caviardé : {output_path}")

        return FileResponse(
            output_path,
            filename=os.path.basename(output_path),
            media_type="application/pdf",
        )

    except Exception as exc:  # pylint: disable=broad-except
        log("❌ Exception pendant la redaction")
        log(traceback.format_exc())
        return PlainTextResponse(f"Internal error: {exc}", status_code=500)
