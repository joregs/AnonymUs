FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app and frontend folders
COPY app/ app/
COPY templates/ templates/
COPY static/ static/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
