# app/rag/textutils.py

import re

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")  # keeps \n, \r and \t


def clean_text(text: str) -> str:
    """Make extracted text safe for tokenizers, JSON and the database."""
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")  # drops lone surrogates
    return _CONTROL.sub(" ", text)