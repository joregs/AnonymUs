import requests

url = "https://nginx.kube.isc.heia-fr.ch/redact/compute"

file_path = "pdfs/in.pdf"
word_to_redact = "Jorge"

with open(file_path, "rb") as f:
    files = {"file": f}
    data = {"word": word_to_redact}
    response = requests.post(url, files=files, data=data)
    # print("Status code:", response.status_code)
    if response.status_code == 200 and response.headers["Content-Type"] == "application/pdf":
        with open("in_redacted_from_k8s.pdf", "wb") as out_file:
            out_file.write(response.content)
        print("PDF censuré téléchargé sous 'in_redacted_from_k8s.pdf'")
    else:
        print("Erreur :", response.text)
