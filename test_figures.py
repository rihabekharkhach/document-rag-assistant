import sys

from app.rag.figures import extract_figures

for f in extract_figures(sys.argv[1]):
    print(f"p{f.page} | {f.image_path} | {f.caption[:80]!r}")