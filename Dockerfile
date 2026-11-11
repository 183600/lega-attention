# LEGA Attention - Reproducible Research Environment
#
# Build: docker build -t lega-attention:0.1.0 .
# Run:   docker run --rm -it lega-attention:0.1.0 bash
#
# For GPU: docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
#          -t lega-attention:0.1.0-cuda .
#          docker run --gpus all --rm -it lega-attention:0.1.0-cuda bash
#
# Reproducibility contract (enforced by scripts/check_requirements_lock.py and
# tests/test_docker_repro.py, not just by convention):
#   * The runtime stack is installed from requirements-lock.txt only. The torch
#     version inside the image is *derived from that file*, never hard-coded and
#     never pulled with `--upgrade`; an image whose stack drifts from the lock
#     is a different experiment environment and must not be called reproducible.
#   * The build runs `python -m lega.repro --check-env` - the very same
#     fail-closed environment gate the pipeline applies to formal runs. A
#     mismatch fails the build instead of silently shipping a container that
#     the pipeline would later refuse to use.
#   * The base image is passed in as ARG so a formal reproduction can pin it by
#     digest (--build-arg PYTHON_IMAGE=python:3.12-slim@sha256:...). The default
#     floating tag is meant for local development only.
#
# The GPU stage replaces the CPU wheel with the CUDA wheel of the *same locked
# version* (local build tag +cu126 is ignored when comparing public versions).
# TORCH_INDEX_URL selects the CUDA wheel index; it must publish the locked
# version, otherwise the build fails loudly instead of silently substituting a
# different torch.

ARG PYTHON_VERSION=3.12
ARG PYTHON_IMAGE="python:${PYTHON_VERSION}-slim"
ARG CPU_TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

FROM ${PYTHON_IMAGE} AS base

ARG CPU_TORCH_INDEX_URL

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

# Runtime dependencies: exactly the locked versions. torch comes from the CPU
# wheel index (the +cpu build referenced by requirements-lock.txt); the other
# packages are installed from the lock file itself. CI/test tooling is NOT part
# of the runtime environment identity and lives in the dev stage.
RUN set -eu; \
    TORCH_PIN="$(grep -E '^torch==' requirements-lock.txt | head -n 1 | sed -n 's/^torch==//p' | tr -d '[:space:]')"; \
    if [ -z "${TORCH_PIN}" ]; then \
        echo "requirements-lock.txt does not pin a torch version" >&2; \
        exit 1; \
    fi; \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel; \
    python -m pip install --no-cache-dir "torch==${TORCH_PIN}" --index-url "${CPU_TORCH_INDEX_URL}"; \
    python -m pip install --no-cache-dir -r requirements-lock.txt

# Copy source code
COPY . .

# Install package in development mode
RUN python -m pip install --no-cache-dir -e .

# Build-time environment gate: the image must satisfy requirements-lock.txt.
# Same gate as the pipeline's formal-run check, so a container that cannot run
# the experiments never reaches a user.
RUN python -m lega.repro --check-env

# Verify installation
RUN python -c "import lega; print(f'LEGA version: {lega.__version__}')"

# Default command
CMD ["bash"]


FROM base AS gpu

ARG TORCH_INDEX_URL=""

# Swap the CPU wheel for the CUDA wheel of the same locked version. The version
# is read from requirements-lock.txt (never hard-coded), so the GPU image and
# the CPU image cannot disagree about which torch the experiment used.
RUN set -eu; \
    TORCH_PIN="$(grep -E '^torch==' requirements-lock.txt | head -n 1 | sed -n 's/^torch==//p' | tr -d '[:space:]')"; \
    if [ -z "${TORCH_PIN}" ]; then \
        echo "requirements-lock.txt does not pin a torch version" >&2; \
        exit 1; \
    fi; \
    TORCH_PIN="${TORCH_PIN%%+*}"; \
    python -m pip uninstall -y torch; \
    if [ -n "${TORCH_INDEX_URL}" ]; then \
        python -m pip install --no-cache-dir "torch==${TORCH_PIN}" --index-url "${TORCH_INDEX_URL}"; \
    else \
        python -m pip install --no-cache-dir "torch==${TORCH_PIN}"; \
    fi; \
    python -m lega.repro --check-env

# Verify CUDA availability
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"


FROM base AS dev

# Development / test tooling: pinned by requirements-ci.txt, kept out of the
# runtime environment so the research image stays exactly the locked stack.
RUN python -m pip install --no-cache-dir -r requirements-ci.txt && \
    python -m pip install --no-cache-dir \
    ipython \
    ipdb \
    jupyter \
    pre-commit

# Install pre-commit hooks
RUN pre-commit install || true
