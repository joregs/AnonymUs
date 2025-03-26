import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import os

def extract_text_with_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text.strip(), doc

def convert_pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)

def redact_text_on_image(image, old_text):
    ocr_data = pytesseract.image_to_data(image, lang='eng+fra', output_type=pytesseract.Output.DICT)
    
    draw = ImageDraw.Draw(image)
    for i, word in enumerate(ocr_data["text"]):
        if old_text.lower() in word.lower():
            (x, y, w, h) = (ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i])
            draw.rectangle([x, y, x + w, y + h], fill="black")

    return image

def modify_pdf(pdf_path, old_text):
    extracted_text, doc = extract_text_with_pymupdf(pdf_path)
    
    if extracted_text:
        # Le PDF contient du texte sélectionnable
        output_pdf_path = pdf_path.replace(".pdf", "_redacted.pdf")
        for page in doc:
            text_instances = page.search_for(old_text)
            for rect in text_instances:
                page.add_redact_annot(rect, fill=(0, 0, 0))
            page.apply_redactions()
        doc.save(output_pdf_path)
        doc.close()
        print(f"PDF texte modifié sauvegardé sous {output_pdf_path}")
    else:
        # Le PDF est un scan, traiter avec OCR
        images = convert_pdf_to_images(pdf_path)
        redacted_images = [redact_text_on_image(img, old_text) for img in images]
        output_pdf_path = pdf_path.replace(".pdf", "_redacted.pdf")
        redacted_images[0].save(output_pdf_path, save_all=True, append_images=redacted_images[1:])
        print(f"PDF image modifié sauvegardé sous {output_pdf_path}")
    
    return output_pdf_path
