# Training / evaluation image (GPU).
#
# The game itself is never baked in: every environment opens the official
# HTML5 QWOP page from foddy.net at runtime and reuses that page.
ARG BASE_IMAGE=pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime
FROM ${BASE_IMAGE}

# The base image installs PyTorch into the system Python, so we do the same.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/

# Tesseract's English data for reading the distance on the canvas
# (qwop_agent/vision.py). The OCR engine itself ships with the tesserocr wheel.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY qwop_agent ./qwop_agent
RUN pip install ".[train]" \
 && playwright install --with-deps --only-shell chromium \
 && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-deps .
ENTRYPOINT ["qwop-agent"]
CMD ["--help"]
