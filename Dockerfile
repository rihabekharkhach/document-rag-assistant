# syntax=docker/dockerfile:1.7

# ---------- stage 1: turn poetry.lock into pinned requirements ----------
FROM python:3.11-slim AS exporter
RUN pip install --no-cache-dir poetry poetry-plugin-export
WORKDIR /w
COPY pyproject.toml poetry.lock ./
RUN poetry export --only main --without-hashes -o requirements.txt \
 # NVIDIA GPU libraries are not needed: we install the CPU build of PyTorch below
 && grep -v -E "^(nvidia-|triton)" requirements.txt > requirements.api.txt \
 && grep -m1 -E "^torch==" requirements.txt | sed -E "s/^torch==([^ ;]+).*/\1/" > torch.version

# ---------- API ----------
FROM python:3.11-slim AS api
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=exporter /w/requirements.api.txt /w/torch.version ./
# the same torch version as in poetry.lock, but the small CPU-only build
RUN pip install "torch==$(cat torch.version)" --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.api.txt
COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- UI ----------
FROM python:3.11-slim AS ui
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.ui.txt .
RUN pip install -r requirements.ui.txt
COPY .streamlit ./.streamlit
COPY ui ./ui
EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]