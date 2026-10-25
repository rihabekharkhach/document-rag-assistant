# app/rag/ocr.py

"""OCR for figures: image file -> cleaned text (labels, axis names, numbers)."""

import os

import pytesseract
from dotenv import load_dotenv
from PIL import Image, ImageOps

load_dotenv()

if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")

OCR_LANGS = os.getenv("OCR_LANGS", "eng+fra")
MIN_CONF = 40   # drop words Tesseract itself is unsure about
UPSCALE = 2     # chart text is small: enlarging it makes OCR much more reliable


def ocr_image(image_path: str) -> str:
    image = Image.open(image_path).convert("L")
    image = image.resize((image.width * UPSCALE, image.height * UPSCALE), Image.LANCZOS)
    image = ImageOps.autocontrast(image)

    # psm 11 = "sparse text": suited to charts and diagrams, where text is scattered
    data = pytesseract.image_to_data(
        image, lang=OCR_LANGS, config="--psm 11", output_type=pytesseract.Output.DICT
    )

    lines: dict[tuple, list[str]] = {}
    for i, word in enumerate(data["text"]):
        word = word.strip()
        try:
            conf = float(data["conf"][i])
        except ValueError:
            continue
        if not word or conf < MIN_CONF:
            continue
        if len(word) < 2 and not word.isdigit():  # stray single characters are mostly noise
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)

    return "\n".join(" ".join(words) for words in lines.values())