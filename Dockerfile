# base with cuda
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Keep it reproducible + non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Micromamba defaults
ENV MAMBA_ROOT_PREFIX=/opt/conda \
    MAMBA_DOCKERFILE_ACTIVATE=1 \
    PATH=/opt/conda/bin:$PATH

# OS deps + micromamba (from the official endpoint)
# Note: the archive is bzip2-compressed; install bzip2 so tar -xvj works.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl bzip2 bash tini git \
 && curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba \
 && mkdir -p /opt/conda \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy env early to leverage layer caching
COPY environment.yml /tmp/environment.yml

# Create/populate the base env from environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
 && micromamba clean --all --yes


 RUN apt-get update && \
 apt-get install -y --no-install-recommends build-essential clang && \
 rm -rf /var/lib/apt/lists/*

# make Triton happy + writable cache
ENV CC=/usr/bin/clang \
 CXX=/usr/bin/clang++ \
 TRITON_CACHE_DIR=/tmp/triton-cache
RUN mkdir -p /tmp/triton-cache && chmod -R 777 /tmp/triton-cache

# Copy the rest of the app (do this after env to keep cache hits)
COPY . /app
# Ensure your entrypoint is executable
RUN chmod +x /app/entrypoint.sh

# Use tini as PID 1, and run inside the env without relying on 'activate'
ENTRYPOINT ["/usr/bin/tini", "--", "micromamba", "run", "-n", "base", "/app/entrypoint.sh"]

# Default args for entrypoint (overridable)
CMD ["--mode=train"]
