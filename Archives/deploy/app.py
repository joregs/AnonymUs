# app.py
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

app = Flask(__name__)


model_path = "./french_ner_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

@app.route("/anonymize", methods=["POST"])
def anonymize():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in JSON"}), 400

    text = data["text"]
    entities = ner_pipeline(text)
    words_to_anonymize = [e["word"] for e in entities if e["entity_group"] in ("PER", "ORG")]

    return jsonify({"anonymize": words_to_anonymize})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
