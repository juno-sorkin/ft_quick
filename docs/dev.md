# Developer Guide

This guide provides instructions for setting up the development environment and running the project locally.

## Environment Setup

This project uses `micromamba` for environment management.

1.  **Install micromamba**: Follow the official instructions to install `micromamba` on your system.

2.  **Create the environment**:
    ```bash
    micromamba env create -f environment.yml
    ```

3.  **Activate the environment**:
    ```bash
    micromamba activate text-style-mimicry
    ```

## Running Locally

1.  **Install Dependencies**: Ensure all dependencies from `environment.yml` are installed.

2.  **Set Environment Variables**: Create a `.env` file in the root directory and add the following variables:
    ```
    WANDB_API_KEY=your_wandb_api_key
    AWS_ACCESS_KEY_ID=your_aws_access_key
    AWS_SECRET_ACCESS_KEY=your_aws_secret_key
    AWS_REGION=your_aws_region
    ```

3.  **Run the main script**:
    ```bash
    python src/main.py --mode=train
    ```

## Running Tests

To run the test suite, use `pytest`:

```bash
pytest
```

This will discover and run all tests in the `tests/` directory. The tests are designed to be run on a CPU and should complete quickly.
