import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import os
from datetime import datetime


def extract_text_with_pymupdf(pdf_path):
    print(f"[{datetime.now().isoformat()}] 🔍 Extracting text with PyMuPDF from: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text.strip(), doc


def convert_pdf_to_images(pdf_path):
    print(f"[{datetime.now().isoformat()}] 🖼️ Converting PDF to images for OCR redaction...")
    return convert_from_path(pdf_path)


def redact_text_on_image(image, text_to_redact):
    print(f"[{datetime.now().isoformat()}] 🕵️ Performing OCR on image to redact: {text_to_redact}")
    ocr_data = pytesseract.image_to_data(image, lang='eng+fra', output_type=pytesseract.Output.DICT)
    draw = ImageDraw.Draw(image)

    for i, word in enumerate(ocr_data["text"]):
        if any(word.lower() == target.lower() for target in text_to_redact):
            x, y, w, h = (ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i])
            draw.rectangle([x, y, x + w, y + h], fill="black")
            print(f"    ➤ Redacted word '{word}' at [{x},{y},{w},{h}]")

    return image


def modify_pdf(pdf_path, text_to_redact):
    print(f"\n[{datetime.now().isoformat()}] 🚀 Starting PDF modification")
    print(f"    ➤ Input file: {pdf_path}")
    print(f"    ➤ Raw words string: '{text_to_redact}'")

      # --- 1. Normaliser en liste ------------------------------------------
    if isinstance(text_to_redact, list):
        text_to_redact_list = [w.strip() for w in text_to_redact if w.strip()]
    else:  # str ou autre
        text_to_redact_list = [
            w.strip() for w in str(text_to_redact).replace(";", ",").split(",") if w.strip()
        ]

    print(f"[{datetime.now().isoformat()}] 🚀 Starting PDF modification")
    print(f"    ➤ Input file: {pdf_path}")
    print(f"    ➤ Parsed redact list: {text_to_redact_list}")

    extracted_text, doc = extract_text_with_pymupdf(pdf_path)
    output_pdf_path = pdf_path.replace(".pdf", "_redacted.pdf")

    if extracted_text:
        print(f"[{datetime.now().isoformat()}] 🧾 PDF contains selectable text")
        for page_number, page in enumerate(doc, start=1):
            for word in text_to_redact_list:
                text_instances = page.search_for(word)
                if text_instances:
                    print(f"    ➤ Found '{word}' on page {page_number}: {len(text_instances)} match(es)")
                for rect in text_instances:
                    page.add_redact_annot(rect, fill=(0, 0, 0))
            page.apply_redactions()
        doc.save(output_pdf_path)
        doc.close()
        print(f"[{datetime.now().isoformat()}] ✅ PDF texte modifié sauvegardé sous {output_pdf_path}")
    else:
        print(f"[{datetime.now().isoformat()}] 📷 PDF appears to be image-based — using OCR")
        images = convert_pdf_to_images(pdf_path)
        redacted_images = [redact_text_on_image(img, text_to_redact_list) for img in images]
        redacted_images[0].save(output_pdf_path, save_all=True, append_images=redacted_images[1:])
        print(f"[{datetime.now().isoformat()}] ✅ PDF image modifié sauvegardé sous {output_pdf_path}")

    return output_pdf_path
