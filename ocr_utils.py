# ocr_utils.py

import easyocr
import fitz  # PyMuPDF
import numpy as np

# Initialize once
reader = easyocr.Reader(['en'])

def ocr_fallback(uploaded_pdf):
    """
    1) Load PDF from bytes with PyMuPDF
    2) Render each page to a NumPy array
    3) OCR each image with EasyOCR
    """
    # Read uploaded bytes
    pdf_bytes = uploaded_pdf.read()

    # Open in-memory PDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    all_text = []
    for page in doc:
        # Render page to an RGB pixmap
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        # Convert to NumPy array (H x W x 3)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        # OCR this page
        words = reader.readtext(arr, detail=0)
        all_text.append(" ".join(words))

    doc.close()
    return "\n\n".join(all_text)