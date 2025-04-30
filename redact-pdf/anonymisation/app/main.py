print(">>> ✅ FastAPI launched with root_path = /anonymisation")

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from fastapi.responses import JSONResponse
import nltk
from nltk.corpus import stopwords
import string

# Init app
app = FastAPI(root_path="/anonymisation")

# Load model
model_path = "./app/frenchNerModel"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Download stopwords
nltk.download("stopwords")
FRENCH_STOPWORDS = set(stopwords.words("french"))

# Helper to clean up
def clean_words(words, stopwords_set):
    return [
        word for word in words
        if word.strip() and
           len(word) > 2 and
           word.lower() not in stopwords_set and
           not all(char in string.punctuation for char in word)
    ]

# Request model
class TextInput(BaseModel):
    text: str

@app.post("/compute")
async def anonymize_text(input_data: TextInput):
    text = input_data.text
    entities = ner_pipeline(text)

    raw_words = [e["word"] for e in entities if e["entity_group"] in ("PER", "ORG", "LOC")]
    filtered_words = clean_words(raw_words, FRENCH_STOPWORDS)

    return {"anonymize": filtered_words}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
