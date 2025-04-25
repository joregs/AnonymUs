import requests

# URLs des microservices
EXTRACT_URL = "https://nginx.kube.isc.heia-fr.ch/extract-text/compute"
REDACT_URL = "https://nginx.kube.isc.heia-fr.ch/redact-pdf/compute"

file_path = "../pdfs/in.pdf"
word_to_redact = "Jorge"

with open(file_path, "rb") as f:
    redact_response = requests.post(REDACT_URL, files={"file": f}, data={"word": "Jorge, Machado"})

if redact_response.status_code == 200 and redact_response.headers["Content-Type"] == "application/pdf":
    with open("in_redacted_from_k8s.pdf", "wb") as out_file:
        out_file.write(redact_response.content)
    print("✅ PDF censuré téléchargé sous 'in_redacted_from_k8s.pdf'")
else:
    print("❌ Erreur de censure :", redact_response.text)
