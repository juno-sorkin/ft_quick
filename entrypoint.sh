#!/bin/bash
set -e

# Default mode if not set
MODE=${MODE:-train}

echo "--- Starting Entrypoint Script ---"
echo "MODE: $MODE"

# 1. Sync data from S3 (Placeholder)
echo "--- Syncing data from S3 ---"
# This is a placeholder. In a real scenario, you would use aws s3 sync.
# Example: aws s3 sync s3://${S3_INPUT_BUCKET}/ /app/data_box/inputs/
echo "Simulating S3 data download..."
mkdir -p /app/data_box/inputs
touch /app/data_box/inputs/placeholder_train.jsonl

# 2. Run the main Python script
echo "--- Executing main application ---"
python src/main.py "$@"

# 3. Sync artifacts to S3 (Placeholder)
echo "--- Syncing artifacts to S3 ---"
# This is a placeholder. In a real scenario, you would use aws s3 sync.
# Example: aws s3 sync /app/data_box/outputs/ s3://${S3_OUTPUT_BUCKET}/
echo "Simulating S3 artifact upload..."
touch /app/data_box/outputs/placeholder_log.txt

echo "--- Entrypoint Script Finished ---"
