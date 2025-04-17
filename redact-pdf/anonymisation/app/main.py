print(">>> ✅ FastAPI launched with root_path = /anonymisation")

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from fastapi.responses import JSONResponse

app = FastAPI(root_path="/anonymisation")

# Load model
model_path = "./app/frenchNerModel"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Request model
class TextInput(BaseModel):
    text: str

@app.post("/compute")
async def anonymize_text(input_data: TextInput):
    text = input_data.text
    entities = ner_pipeline(text)
    words_to_anonymize = [e["word"] for e in entities if e["entity_group"] in ("PER","ORG","LOC")]
    return {"anonymize": words_to_anonymize}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
