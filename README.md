# Text Style Mimicry with unsloth/gpt-oss-20b

This project aims to fine-tune the `unsloth/gpt-oss-20b` model to mimic a specific text style. It utilizes the Hugging Face ecosystem, PEFT for LoRA fine-tuning, and Unsloth for performance optimization.

## Project Structure

- **`.github/workflows/`**: Contains the CI/CD pipelines for building, testing, releasing, and deploying the model.
- **`config/`**: Contains configuration files, such as `config.yml` for hyperparameters and settings.
- **`data_box/`**: A directory for local data handling.
    - **`inputs/`**: For training and validation data.
    - **`outputs/`**: For storing logs, trained model adapters, and other artifacts.
- **`docs/`**: Project documentation.
- **`src/`**: Source code for the fine-tuning process.
- **`tests/`**: Pytest test suite.

## Usage

1. **Build the Docker image:**
   ```bash
   docker build -t text-style-mimicry .
   ```
2. **Run the training script:**
   ```bash
   docker run --gpus all -v $(pwd)/data_box/outputs:/app/data_box/outputs text-style-mimicry
   ```

## CI/CD

The project includes two GitHub Actions workflows:

- **`btr.yml`**: Builds the Docker image, runs tests, and releases the image to Amazon ECR on every push to the `main` branch.
- **`deploy.yml`**: Manually triggered workflow to deploy and run the training job on an EC2 instance.

Refer to the documentation in the `docs` directory for more details.
