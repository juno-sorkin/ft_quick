from datasets import load_dataset
import os

def load_and_prepare_dataset(data_config, tokenizer, paths_config):
    """
    Loads a dataset from a local path and prepares it for training.

    Args:
        data_config (dict): Configuration for data loading.
        tokenizer: The tokenizer to use for processing the text.
        paths_config (dict): Configuration for file paths.

    Returns:
        A processed dataset ready for training.
    """
    print("--- Loading and preparing dataset ---")

    # Construct file paths
    train_file_path = os.path.join(paths_config['input_dir'], data_config['train_file'])
    validation_file_path = os.path.join(paths_config['input_dir'], data_config['validation_file'])

    # Check if files exist
    if not os.path.exists(train_file_path):
        raise FileNotFoundError(f"Train file not found at: {train_file_path}")
    if not os.path.exists(validation_file_path):
        raise FileNotFoundError(f"Validation file not found at: {validation_file_path}")

    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file_path,
            "validation": validation_file_path,
        },
    )

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples[data_config['dataset_text_field']],
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding="max_length",
        )

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names, # Remove original columns
    )

    print("--- Dataset loaded and prepared successfully ---")
    return tokenized_dataset
