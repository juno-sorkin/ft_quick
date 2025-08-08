#!/bin/bash
set -e

# Default mode if not set
MODE=${MODE:-train}

echo "--- Starting Entrypoint Script ---"
echo "MODE: $MODE"

# 0. GPU smoke test (hard requirement)
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- GPU Info (nvidia-smi) ---"
  if ! nvidia-smi; then
    echo "Error: nvidia-smi failed. GPU not available or driver/runtime issue."
    exit 1
  fi
else
  echo "Error: nvidia-smi not found. CUDA runtime not available in this environment."
  exit 1
fi

# 1. Sync data from S3 (via boto3)
echo "--- Syncing data from S3 ---"
if [ -z "${S3_BUCKET}" ]; then
    echo "Error: S3_BUCKET environment variable is not set"
    exit 1
fi

# Sync from S3 using boto3 utility
python src/s3_sync.py download --bucket "${S3_BUCKET}" --prefix "inputs/" --dest "/app/data_box/inputs"
if [ $? -ne 0 ]; then
    echo "Error: Failed to sync data from S3"
    exit 1
fi

# 2. Inspect the model architecture
echo "--- Inspecting model architecture ---"
python inspect_arch.py

# 2. Run the main Python script
echo "--- Executing main application ---"
python src/main.py --mode "${MODE}"

# 3. Sync artifacts to S3 (via boto3)
echo "--- Syncing artifacts to S3 ---"
if [ -z "${S3_BUCKET}" ]; then
    echo "Error: S3_BUCKET environment variable is not set"
    exit 1
fi

# Sync to S3 using boto3 utility
python src/s3_sync.py upload --src "/app/data_box/outputs" --bucket "${S3_BUCKET}" --prefix "outputs/"
if [ $? -ne 0 ]; then
    echo "Error: Failed to sync artifacts to S3"
    exit 1
fi

echo "--- Entrypoint Script Finished ---"
