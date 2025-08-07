# CI/CD Pipeline

This document outlines the Continuous Integration/Continuous Deployment (CI/CD) pipeline for the project.

## 1. Build, Test, Release (`btr.yml`)

This workflow is triggered on every push or merge to the `main` branch.

### Steps:

1.  **Authenticate with AWS ECR**: Pulls cached Docker layers to speed up the build process.
2.  **Build Docker Image**: Builds the application Docker image using the `Dockerfile`.
3.  **Run Local Tests**: Executes the `pytest` suite to perform smoke tests and ensure the model runs on a CPU.
4.  **Release to ECR**: If tests pass, the newly built Docker image is tagged and pushed to the Amazon Elastic Container Registry (ECR).

## 2. Deploy (`deploy.yml`)

This workflow is triggered manually via the GitHub Actions UI or CLI.

### Steps:

1.  **Assume AWS Role**: Assumes the necessary IAM role for managing AWS resources.
2.  **Start EC2 Instance**: Increases the desired capacity of the Auto Scaling Group (ASG) to 1 to launch a GPU-enabled EC2 instance.
3.  **Wait for Instance**: Waits for the instance to be in a running and healthy state.
4.  **Run Training Job via SSM**:
    *   Uses AWS Systems Manager (SSM) Run Command to execute a shell script on the instance.
    *   The script first runs a smoke test to verify `nvidia-smi` and CUDA are available.
    *   It then starts the Docker container with `--gpus all`.
    *   Environment variables (like W&B keys) are injected from GitHub Secrets.
    *   The container's entrypoint script pulls data from an S3 bucket.
    *   The main training script (`src/main.py`) is executed.
    *   Logs, checkpoints, and model adapters are saved to a local path inside the container, which is then synced to an S3 bucket.
5.  **Shutdown EC2 Instance**: Decreases the desired capacity of the ASG back to 0 to terminate the instance and save costs.
