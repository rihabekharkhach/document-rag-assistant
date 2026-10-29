# app/rag/sparse.py

"""Keyword (sparse) vectors: tokens hashed to integer ids, weighted by sublinear term frequency.
Qdrant applies IDF at query time (collection configured with Modifier.IDF)."""

import re
import unicodedata
import zlib
from collections import Counter
from math import log

from qdrant_client.models import SparseVector

TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")  # words, ids like wc9 / g4, decimals like 95.4
INDEX_SPACE = 2**28


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()  # strip accents
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)  # 95,4 -> 95.4
    return TOKEN.findall(text)


def _index(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) % INDEX_SPACE  # stable across runs, unlike hash()


def sparse_document(text: str) -> SparseVector:
    weights: dict[int, float] = {}
    for token, count in Counter(tokenize(text)).items():
        index = _index(token)
        weights[index] = weights.get(index, 0.0) + 1.0 + log(count)
    return SparseVector(indices=list(weights), values=list(weights.values()))


def sparse_query(text: str) -> SparseVector:
    indices = sorted({_index(token) for token in tokenize(text)})
    return SparseVector(indices=indices, values=[1.0] * len(indices))