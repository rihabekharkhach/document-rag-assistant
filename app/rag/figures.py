# app/rag/figures.py

"""
Extract figures from a PDF. Each embedded image is saved twice:
  - image_path:  200 dpi render, used for display and OCR
  - vision_path: render at the figure's native resolution, used by the vision model
Each figure is paired with its caption (the "Figure N: ..." text printed below it).
"""

from dataclasses import dataclass
from pathlib import Path

import pymupdf

MIN_SIDE_PX = 150
MAX_VISION_DPI = 600
MAX_VISION_PIXELS = 12_000_000  # keeps one image at a reasonable size for the API
CAPTION_PREFIXES = ("figure", "fig.", "fig ")


@dataclass
class Figure:
    source: str
    page: int          # 1-based page number
    image_path: str
    vision_path: str
    caption: str
    nearby: str = ""


def _find_caption(blocks: list, rect: pymupdf.Rect, max_gap: float = 60) -> str:
    """Caption = nearest text block starting with 'Figure', just below the image."""
    best, best_gap = "", max_gap + 1
    for x0, y0, x1, y1, text, _block_no, block_type in blocks:
        if block_type != 0:  # 0 = text block, 1 = image block
            continue
        text = " ".join(text.split())
        if not text.lower().startswith(CAPTION_PREFIXES):
            continue
        overlaps_horizontally = x0 < rect.x1 and x1 > rect.x0
        gap = y0 - rect.y1
        if overlaps_horizontally and -5 <= gap <= max_gap and gap < best_gap:
            best, best_gap = text, gap
    return best

def _text_below(blocks: list, rect: pymupdf.Rect, max_gap: float = 90, max_chars: int = 400) -> str:
    """Text printed right below the figure: caption, legend, label order (may include the start of body text)."""
    found = []
    for x0, y0, x1, y1, text, _n, block_type in blocks:
        if block_type != 0:
            continue
        gap = y0 - rect.y1
        if -5 <= gap <= max_gap and x0 < rect.x1 and x1 > rect.x0:
            found.append((y0, " ".join(text.split())))
    found.sort()
    return " ".join(t for _, t in found)[:max_chars]


def _vision_dpi(native_width_px: int, rect: pymupdf.Rect) -> int:
    """DPI that reproduces the embedded image's native resolution, capped for size."""
    inches_w, inches_h = max(rect.width / 72, 0.1), max(rect.height / 72, 0.1)
    dpi = max(200.0, min(native_width_px / inches_w, MAX_VISION_DPI))
    pixels = (inches_w * dpi) * (inches_h * dpi)
    if pixels > MAX_VISION_PIXELS:
        dpi *= (MAX_VISION_PIXELS / pixels) ** 0.5
    return int(round(dpi))


def extract_figures(pdf_path: str, out_dir: str = "data/figures") -> list[Figure]:
    pdf = Path(pdf_path)
    output = Path(out_dir) / pdf.stem
    output.mkdir(parents=True, exist_ok=True)

    figures: list[Figure] = []
    doc = pymupdf.open(pdf_path)

    for page_number, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")
        seen: set[tuple] = set()

        for index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            try:
                native_width = doc.extract_image(xref)["width"]
            except Exception:
                native_width = 0

            for rect in page.get_image_rects(xref):
                key = (xref, round(rect.x0), round(rect.y0))
                if key in seen:
                    continue
                seen.add(key)

                pix = page.get_pixmap(clip=rect, dpi=200)
                if pix.width < MIN_SIDE_PX or pix.height < MIN_SIDE_PX:
                    continue

                path = output / f"p{page_number}_img{index}.png"
                pix.save(str(path))

                vision_path = str(path)
                dpi = _vision_dpi(native_width, rect) if native_width else 200
                if dpi > 210:  # the native image is sharper than the 200 dpi render
                    hi_path = output / f"p{page_number}_img{index}_vision.png"
                    page.get_pixmap(clip=rect, dpi=dpi).save(str(hi_path))
                    vision_path = str(hi_path)

                figures.append(Figure(
                    source=pdf.name,
                    page=page_number,
                    image_path=str(path),
                    vision_path=vision_path,
                    caption=_find_caption(blocks, rect)[:160],
                    nearby=_text_below(blocks, rect),
                ))

    doc.close()
    return figures