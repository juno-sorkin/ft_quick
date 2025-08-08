#!/bin/bash
set -e

# Default mode if not set
MODE=${MODE:-train}

echo "--- Starting Entrypoint Script ---"
echo "MODE: $MODE"

# 1. Sync data from S3
echo "--- Syncing data from S3 ---"
if [ -z "${S3_BUCKET}" ]; then
    echo "Error: S3_BUCKET environment variable is not set"
    exit 1
fi

# Sync from S3
aws s3 sync s3://${S3_BUCKET}/inputs/ /app/data_box/inputs/
if [ $? -ne 0 ]; then
    echo "Error: Failed to sync data from S3"
    exit 1
fi

# 2. Inspect the model architecture
echo "--- Inspecting model architecture ---"
python inspect_arch.py

# 2. Run the main Python script
echo "--- Executing main application ---"
python src/main.py "$@"

# 3. Sync artifacts to S3
echo "--- Syncing artifacts to S3 ---"
if [ -z "${S3_BUCKET}" ]; then
    echo "Error: S3_BUCKET environment variable is not set"
    exit 1
fi

# Sync to S3
aws s3 sync /app/data_box/outputs/ s3://${S3_BUCKET}/outputs/
if [ $? -ne 0 ]; then
    echo "Error: Failed to sync artifacts to S3"
    exit 1
fi

echo "--- Entrypoint Script Finished ---"
