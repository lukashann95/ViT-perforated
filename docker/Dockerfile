# Image for PAI training on an L40 / L40S (or any GPU torch 2.6.0+cu124 supports: sm_80/86/89/90).
# Holds the code and both venvs that scripts/setup_machine.sh would create; the dataset, the keys
# (.env) and the runs stay on the host and are mounted in. Build from the repo root:
#   docker build -f docker/Dockerfile -t vit-perforated-l40 .
# Then use docker/run_l40.sh (see docker/README.md).

# The CUDA runtime comes with the torch wheels; this base only adds the NVIDIA container settings.
# The host needs an NVIDIA driver >= 550 and the NVIDIA Container Toolkit.
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /usr/local/bin/
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Writable HOME and a fixed USER, so the container also runs as the host user (docker run --user):
# uv, W&B and torch keep caches in HOME, and Python's getpass needs USER for an unknown uid.
ENV HOME=/home/app \
    USER=app \
    UV_PYTHON=3.12 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    DATA_DIR=/data \
    PYTHONUNBUFFERED=1
WORKDIR /opt/ViT-perforated

# Training venv from uv.lock (torch 2.6.0+cu124), and the small pyarrow venv for unpacking, at
# the paths setup_machine.sh uses. Made writable by any uid in the same layer (a separate chmod
# would copy the 5 GB venv into a new layer): uv locks the venv when setup_machine.sh checks it.
COPY pyproject.toml uv.lock ./
# The download cache lives in a build cache mount, outside the image and outside HOME.
RUN --mount=type=cache,target=/uv-build-cache \
    export UV_CACHE_DIR=/uv-build-cache \
    && uv sync --frozen \
    && uv venv .venv-data \
    && uv pip install --python .venv-data/bin/python pyarrow \
    && mkdir -p /home/app /data \
    && chmod -R a+rwX /opt/ViT-perforated /opt/uv-python /home/app /data

COPY . .
RUN chmod -R a+rwX /opt/ViT-perforated/*.py /opt/ViT-perforated/scripts

CMD ["bash"]
