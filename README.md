# Text Style Mimicry with unsloth/gpt-oss-20b

this program was written in 1 day to LoRA finetune the new gpt-oss-20b MOE multimodel released 8/5 
-finetines using perplexity based loss curves and recursive hyperparameter tuning to mimic writing style
    in this case I fine tuned on some essays i wrote in highschool pulled from an S3 bucket at runtime
the workflow intentionally utizilizes a scalable CI/CD setup with AWS and Github actions runner, it runs on a ec2 g5.xlarge spot instance via ASG  

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
