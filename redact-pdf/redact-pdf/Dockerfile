FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    poppler-utils tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng \
    && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

