import re
import sys

from app.rag.figures import extract_figures
from app.rag.ocr import ocr_image

EXPECTED = {
    3: ["Motor", "Gearbox"],
    4: ["Task Loss", "Domain Accuracy"],
    5: ["0.31", "0.39"],
    6: ["SpEnt_cur", "Std_RMS_phases", "Ratio_in_out"],
    7: ["Motor", "Gearbox"],
}

for fig in extract_figures(sys.argv[1]):
    text = ocr_image(fig.image_path)
    match = re.match(r"figure\s+(\d+)", fig.caption.lower())
    number = int(match.group(1)) if match else None

    print(f"\n=== p{fig.page} | {fig.caption[:60]} | {len(text)} chars ===")
    for needle in EXPECTED.get(number, []):
        print(f"   {'FOUND  ' if needle.lower() in text.lower() else 'MISSING'} {needle}")
    print(text[:300].replace("\n", " | "))