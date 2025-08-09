# base with cuda
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Better logging in ECS
ENV PYTHONUNBUFFERED=1

# Micromamba defaults
ENV MAMBA_ROOT_PREFIX=/opt/conda \
    MAMBA_DOCKERFILE_ACTIVATE=1 \
    PATH=/opt/conda/bin:$PATH

# OS deps (add compiler toolchain + tini up front)
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl bzip2 bash git git-lfs tini \
      build-essential clang \
 && rm -rf /var/lib/apt/lists/*

# Install micromamba
RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba \
 && mkdir -p /opt/conda

WORKDIR /app

# Create env early to leverage caching
COPY environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
 && micromamba clean --all --yes

# Make Triton happy + writable cache
ENV CC=/usr/bin/clang \
    CXX=/usr/bin/clang++ \
    TRITON_CACHE_DIR=/tmp/triton-cache \
    HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf/transformers \
    HF_DATASETS_CACHE=/cache/hf/datasets
RUN mkdir -p /tmp/triton-cache /cache/hf/transformers /cache/hf/datasets && \
    chmod -R 777 /tmp/triton-cache /cache

# App code last for better layer reuse
COPY . /app
RUN chmod +x /app/entrypoint.sh

# Tini as PID 1 with subreaper (-s)
ENTRYPOINT ["/usr/bin/tini","-s","--","micromamba","run","-n","base","/app/entrypoint.sh"]

# Overridable default
CMD ["--mode=train"]
