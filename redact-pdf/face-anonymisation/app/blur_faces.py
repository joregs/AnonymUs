import fitz
import cv2
import numpy as np
import io
from PIL import Image

def detectAndBlurFaces(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    blurredGlobal = cv2.GaussianBlur(image, (51, 51), 0)
    diff = cv2.absdiff(image, blurredGlobal)
    diffGray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, maskFlou = cv2.threshold(diffGray, 10, 255, cv2.THRESH_BINARY_INV)

    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        roiMask = maskFlou[y:y+h, x:x+w]
        flouRatio = np.sum(roiMask == 255) / (w * h)

        if flouRatio > 0.8:
            continue

        aspect_ratio = w / h
        if 0.75 < aspect_ratio < 1.33:
            roi = image[y:y+h, x:x+w]
            blurred = cv2.GaussianBlur(roi, (99, 99), 30)
            image[y:y+h, x:x+w] = blurred

    return image

def pixmapToImage(pix: fitz.Pixmap) -> np.ndarray:
    if pix.n not in (3, 4):
        raise ValueError("Pixmap n'est pas bon")
    if pix.alpha:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def blur_faces(input_path: str, output_path: str):
    doc = fitz.open(input_path)
    newDoc = fitz.open()

    for pageIndex in range(len(doc)):
        page = doc.load_page(pageIndex)
        pix = page.get_pixmap(dpi=150)

        image = pixmapToImage(pix)
        blurred = detectAndBlurFaces(image)

        rgb_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        width, height = pix.width, pix.height
        pageRect = fitz.Rect(0, 0, width, height)
        newPage = newDoc.new_page(width=width, height=height)
        newPage.insert_image(pageRect, stream=img_bytes.getvalue())

    newDoc.save(output_path)
    newDoc.close()
    doc.close()
