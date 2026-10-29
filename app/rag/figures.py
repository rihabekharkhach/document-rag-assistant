# app/rag/figures.py

"""
Extract figures from a PDF: each embedded image is rendered to a PNG file
and paired with its caption (the "Figure N: ..." text printed below it).
"""

from dataclasses import dataclass
from pathlib import Path

import pymupdf

MIN_SIDE_PX = 150  # ignore logos, icons and decorative bits
CAPTION_PREFIXES = ("figure", "fig.", "fig ")


@dataclass
class Figure:
    source: str
    page: int         # 1-based page number
    image_path: str
    caption: str


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
            for rect in page.get_image_rects(xref):
                key = (xref, round(rect.x0), round(rect.y0))
                if key in seen:
                    continue
                seen.add(key)

                # Render the region of the page where the image sits
                pix = page.get_pixmap(clip=rect, dpi=200)
                if pix.width < MIN_SIDE_PX or pix.height < MIN_SIDE_PX:
                    continue

                path = output / f"p{page_number}_img{index}.png"
                pix.save(str(path))
                figures.append(Figure(
                    source=pdf.name,
                    page=page_number,
                    image_path=str(path),
                    caption=_find_caption(blocks, rect)[:160],
                ))

    doc.close()
    return figures