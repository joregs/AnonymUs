
from __future__ import annotations

import boto3
from botocore.client import Config
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import requests
import time
from requests.exceptions import Timeout, RequestException
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Dict
from enum import Enum
from uuid import UUID
from urllib.parse import urljoin


from pydantic import BaseModel



# ---------------------------------------------------------------------------
# App & templates
# ---------------------------------------------------------------------------
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ---------------------------------------------------------------------------
# Micro‑service endpoints (cluster‑local)
# ---------------------------------------------------------------------------
EXTRACT_URL = os.getenv("EXTRACT_URL", "http://extract-text-service.anonymus.svc.cluster.local/extract-text/compute")
ANONYMISATION_URL = os.getenv("ANONYMISATION_URL", "http://anonymisation-service.anonymus.svc.cluster.local/anonymisation/compute")
REDACT_URL = os.getenv("REDACT_URL", "http://redact-pdf-service.anonymus.svc.cluster.local/redact-pdf/compute")
FACE_ANONYMISATION_URL = os.getenv("FACE_ANONYMISATION_URL", "http://face-anonymisation-service.anonymus.svc.cluster.local/face-anonymisation/compute")
TIMEOUT = 15  # seconds per micro‑service call


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------
class ServiceTaskTask(BaseModel):
    """
    Task update model
    This model is used to update a task
    """
    id: UUID
    data_in: List[str]
    data_out: List[str] | None = None
    status: TaskStatus
    service_id: UUID
    pipeline_execution_id: UUID | None = None


class ServiceTaskBase(BaseModel):
    """
    Base class for Service task
    This model is used in subclasses
    """

    s3_access_key_id: str
    s3_secret_access_key: str
    s3_region: str
    s3_host: str
    s3_bucket: str
    task: ServiceTaskTask
    callback_url: str

class TaskStatus(str, Enum):
    PENDING = "pending"
    FETCHING = "fetching"
    PROCESSING = "processing"
    SAVING = "saving"
    FINISHED = "finished"
    ERROR = "error"
    SCHEDULED = "scheduled"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"


class TaskUpdate(BaseModel):
    """
    Task update model
    This model is used to update a task
    """
    service: str | None = None
    url: str | None = None
    data_out: List[str] | None = None
    status: TaskStatus | None = None

# Routes /status /tasks/taskid/status /compute
TASKS: Dict[UUID, TaskStatus] = {} 
@app.get("/status", response_class=PlainTextResponse)  
async def service_status() -> str:
    return "available"


@app.get("/tasks/{task_id}/status", response_class=PlainTextResponse)  
async def task_status(task_id: UUID) -> str:
    status = TASKS.get(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return status.value

@app.post("/compute", response_class=PlainTextResponse) 
async def compute_task( service_task: ServiceTaskBase, background_tasks: BackgroundTasks) -> str:
    task_id = service_task.task.id
    if task_id in TASKS:
        raise HTTPException(status_code=400, detail="Task already exists")
    # Enqueue
    TASKS[task_id] = TaskStatus.PENDING

    background_tasks.add_task(_process_task, service_task)
    return "Task added to queue"


async def _process_task(service_task: ServiceTaskBase) -> None:
    task_id = service_task.task.id
    session_id = str(task_id)
    session_dir = os.path.join(tempfile.gettempdir(), session_id)
    os.makedirs(session_dir, exist_ok=True)
    print("Starting core engine process...")
    try:
        # -----------------------------------------------------------------
        # 1. FETCHING – récupération des fichiers d'entrée depuis S3
        # -----------------------------------------------------------------
        TASKS[task_id] = TaskStatus.FETCHING
        print("Fetching...")

        s3 = boto3.client(
            "s3",
            aws_access_key_id=service_task.s3_access_key_id,
            aws_secret_access_key=service_task.s3_secret_access_key,
            region_name=service_task.s3_region,
            endpoint_url=service_task.s3_host,
            config=Config(signature_version="s3v4"),
        )

        local_inputs: List[str] = []
        for key in service_task.task.data_in:
            filename = os.path.basename(key)
            local_path = os.path.join(session_dir, filename)
            s3.download_file(service_task.s3_bucket, key, local_path)
            local_inputs.append(local_path)

        # -----------------------------------------------------------------
        # 2. PROCESSING 
        # -----------------------------------------------------------------
        print("Precessing...")

        TASKS[task_id] = TaskStatus.PROCESSING
        outputs: List[str] = []

        for input_path in local_inputs:
            filename = os.path.basename(input_path)
            print("extract-text...")

            # extract-text
            with open(input_path, "rb") as f:
                extract_resp = post_with_retries(
                    EXTRACT_URL,
                    files={"file": (filename, f, "application/pdf")},
                    data={"session_id": session_id},
                    timeout=TIMEOUT,
                )
                
            extract_resp.raise_for_status()
            extracted_text = extract_resp.text

            print("anonymisation...")
            # anonymisation
            anonymise_resp = post_with_retries(
                ANONYMISATION_URL,
                json={"text": extracted_text},
                timeout=TIMEOUT,
            )
            anonymise_resp.raise_for_status()
            to_anonymize: list[str] = anonymise_resp.json().get("anonymize", [])

            if not to_anonymize:
                output_key = f"redacted/{filename}"
                s3.upload_file(input_path, service_task.s3_bucket, output_key)
                outputs.append(output_key)
                continue
            
            print("redact...")

            # redact-pdf
            payload = {
                "session_id": session_id,
                "filename": filename,
                "word": to_anonymize,
            }
            redact_resp = post_with_retries(REDACT_URL, json=payload, timeout=TIMEOUT * 2)
            redact_resp.raise_for_status()
            object_name = redact_resp.json()["filename"]

            print("face anon...")

            # face-anonymisation
            face_payload = {
                "session_id": session_id,
                "filename": object_name,
                "mask_faces": True
            }
            face_resp = post_with_retries(
                FACE_ANONYMISATION_URL, json=face_payload, timeout=TIMEOUT * 2
            )
            face_resp.raise_for_status()

            redacted_local = os.path.join(session_dir, f"redacted_{filename}")
            with open(redacted_local, "wb") as f:
                f.write(face_resp.content)

            # -----------------------------------------------------------------
            # 3. SAVING – upload du PDF final sur S3
            # -----------------------------------------------------------------

            print("Saving...")

            TASKS[task_id] = TaskStatus.SAVING
            output_key = f"{uuid.uuid4()}_{filename}"
            s3.upload_file(redacted_local, service_task.s3_bucket, output_key)
            outputs.append(output_key)
            print(output_key)

            print("Liste des fichiers dans le bucket redacted/")
            response = s3.list_objects_v2(Bucket=service_task.s3_bucket)
            for obj in response.get("Contents", []):
                print("-", obj["Key"])

        # -----------------------------------------------------------------
        # 4. FINISHED – callback et mise à jour du task object
        # -----------------------------------------------------------------
        print("Finishings...")
        
        service_task.task.data_out = outputs
        TASKS[task_id] = TaskStatus.FINISHED

        try:
            print("outputs :")
            print(outputs)

            print(str(service_task.task.service_id))


            update = TaskUpdate(
                service=str(service_task.task.service_id),       
                url="https://anonymus.kube.isc.heia-fr.ch",       
                data_out=outputs,                                 
                status=TaskStatus.FINISHED,
            )

            print(update.dict(exclude_none=True))
            resp = requests.patch(
                service_task.callback_url,
                json=update.dict(exclude_none=True),
                timeout=10,
            )
            print("PATCH : ", resp.status_code, resp.text)
            resp.raise_for_status()
        except Exception as e:
            print("error lors du patch correct:", str(e))

    except Exception as exc:
        TASKS[task_id] = TaskStatus.ERROR

        print("Pipeline error:")

        # TODO : change this to taskupdate
        try:
            error_payload = {
                "status": TaskStatus.ERROR.value,   
                # "data_out": outputs if outputs else None,
                # "service": str(service_task.task.service_id),
                # "url": settings.SERVICE_PUBLIC_URL,
            }

            print("sending error:")

            requests.patch(
                service_task.callback_url,
                json=error_payload,
                timeout=10,
            )
        except Exception:
            print("error lors du patch d'erreur")
        raise



def post_with_retries(url, **kwargs):
    max_retries = 2
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, **kwargs)
            response.raise_for_status()
            return response
        except Timeout:
            print(f"[Timeout] Tentative {attempt}/{max_retries} échouée pour {url}")
            if attempt == max_retries:
                print(f"[Erreur] Service {url} inaccessible après {max_retries} tentatives.")
                raise
            time.sleep(2)


# ---------------------------------------------------------------------------
# WEBAPP
# ---------------------------------------------------------------------------
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/pipeline")
async def process_pipeline(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mask_faces: bool = Form(False),
    mask_person_names: bool = Form(False),
    mask_organizations: bool = Form(False),
    mask_locations: bool = Form(False),
    mask_emails: bool = Form(False),
    mask_phones: bool = Form(False),
    mask_dates: bool = Form(False),
    mask_other: bool = Form(False),
):

    session_id = str(uuid.uuid4())
    session_dir = os.path.join(tempfile.gettempdir(), session_id)
    os.makedirs(session_dir, exist_ok=True)

    # 0. Sauvegarde du fichier d’entrée
    input_path = os.path.join(session_dir, file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1. Extraction de texte
    try:
        with open(input_path, "rb") as f:
            extract_resp = post_with_retries(
                EXTRACT_URL,
                files={"file": (file.filename, f, "application/pdf")},
                data={"session_id": session_id},
                timeout=TIMEOUT,
            )
    except Timeout:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Le service d’extraction a mis trop de temps à répondre (timeout)."
            },
            status_code=504
        )
    # Si post_with_retries renvoie un autre RequestException, il remonte → 500

    extracted_text = extract_resp.text

    # 2. Anonymisation
    mask_options: list[str] = []
    if mask_faces:           mask_options.append("faces")
    if mask_person_names:    mask_options.append("person_names")
    if mask_organizations:   mask_options.append("organizations")
    if mask_locations:       mask_options.append("locations")
    if mask_emails:          mask_options.append("emails")
    if mask_phones:          mask_options.append("phones")
    if mask_dates:           mask_options.append("dates")
    if mask_other:           mask_options.append("other")

    try:
        anonymise_resp = post_with_retries(
            ANONYMISATION_URL,
            json={"text": extracted_text, "mask": mask_options},
            timeout=TIMEOUT,
        )
    except Timeout:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Le service d’anonymisation a mis trop de temps à répondre (timeout)."
            },
            status_code=504
        )

    to_anonymize: list[str] = anonymise_resp.json().get("anonymize", [])

    # Si rien à anonymiser, on retourne directement le PDF d’entrée
    if not to_anonymize:
        background_tasks.add_task(shutil.rmtree, session_dir, ignore_errors=True)
        return FileResponse(input_path, filename=file.filename, media_type="application/pdf")

    # 3. Redaction PDF
    payload_redact = {
        "session_id": session_id,
        "filename": file.filename,
        "word": to_anonymize,
    }
    try:
        redact_resp = post_with_retries(
            REDACT_URL,
            json=payload_redact,
            timeout=TIMEOUT * 2,
        )
    except Timeout:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Le service de redaction PDF a mis trop de temps à répondre (timeout)."
            },
            status_code=504
        )

    object_name = redact_resp.json()["filename"]

    # 4. Face-anonymisation
    payload_face = {
        "session_id": session_id,
        "filename": object_name,
        "mask_faces": mask_faces,
    }
    try:
        face_anon_resp = post_with_retries(
            FACE_ANONYMISATION_URL,
            json=payload_face,
            timeout=TIMEOUT * 2,
        )
    except Timeout:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Le service de face-anonymisation a mis trop de temps à répondre (timeout)."
            },
            status_code=504
        )

    # On enregistre localement le PDF final
    redacted_path = os.path.join(session_dir, f"redacted_{file.filename}")
    with open(redacted_path, "wb") as f_out:
        f_out.write(face_anon_resp.content)

    # 5. Sauvegarde + réponse
    background_tasks.add_task(shutil.rmtree, session_dir, ignore_errors=True)
    return FileResponse(
        redacted_path,
        filename=f"redacted_{file.filename}",
        media_type="application/pdf",
    )