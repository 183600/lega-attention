# LEGA Attention - Reproducible Research Environment
# 
# Build: docker build -t lega-attention:0.1.0 .
# Run:   docker run --rm -it lega-attention:0.1.0 bash
# 
# For GPU: docker build --build-arg CUDA_VERSION=12.1 -t lega-attention:0.1.0-cuda12.1 .
#          docker run --gpus all --rm -it lega-attention:0.1.0-cuda12.1 bash

ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION=""
ARG PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

FROM python:${PYTHON_VERSION}-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set UTF-8 locale
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUTF8=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml requirements.txt requirements-lock.txt requirements-ci.txt ./

# Install Python dependencies (locked versions for reproducibility)
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r requirements-lock.txt && \
    python -m pip install --no-cache-dir -r requirements-ci.txt

# Copy source code
COPY . .

# Install package in development mode
RUN python -m pip install --no-cache-dir -e .

# Verify installation
RUN python -c "import lega; print(f'LEGA version: {lega.__version__}')"

# Default command
CMD ["bash"]


FROM base AS gpu

ARG CUDA_VERSION=12.1
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu${CUDA_VERSION/./}

# Reinstall PyTorch with CUDA support
RUN python -m pip install --no-cache-dir --upgrade \
    torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}

# Verify CUDA availability
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"


FROM base AS dev

# Additional development tools
RUN python -m pip install --no-cache-dir \
    ipython \
    ipdb \
    jupyter \
    pre-commit

# Install pre-commit hooks
RUN pre-commit install || true