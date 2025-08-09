from datasets import Dataset, DatasetDict
import os
import json

def load_and_prepare_dataset(data_config, tokenizer, paths_config=None):
    """
    Loads a dataset from local files and prepares it for training.
    It now handles conversational JSON format by applying a chat template.

    Args:
        data_config (dict): Configuration for data loading.
        tokenizer: The tokenizer to use for processing the text.
        paths_config (dict, optional): Configuration for file paths. Defaults to {'input_dir': 'data_box/inputs'}.

    Returns:
        A processed dataset ready for training.
    """
    print("--- Loading and preparing dataset ---")

    # Set default paths if not provided
    if paths_config is None:
        paths_config = {'input_dir': 'data_box/inputs'}
    if data_config is None:
        data_config = {
            'train_file': 'train.jsonl',
            'validation_file': 'val.jsonl',
            'test_file': 'test.jsonl',
        }

    # Construct file paths
    train_file_path = os.path.join(paths_config['input_dir'], data_config['train_file'])
    validation_file_path = os.path.join(paths_config['input_dir'], data_config['validation_file'])
    test_file_path = os.path.join(paths_config['input_dir'], data_config['test_file'])

    # Helper function to load and process a single jsonl file
    def process_file(file_path):
        formatted_texts = []
        with open(file_path, 'r') as f:
            for line in f:
                messages = json.loads(line)
                formatted_texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
        return formatted_texts

    # Load and process each file
    train_texts = process_file(train_file_path)
    validation_texts = process_file(validation_file_path)
    test_texts = process_file(test_file_path)

    # Create a Hugging Face Dataset
    dataset = Dataset.from_dict({
        'text': train_texts
    })
    validation_dataset = Dataset.from_dict({
        'text': validation_texts
    })
    test_dataset = Dataset.from_dict({
        'text': test_texts
    })
    
    dataset = DatasetDict({
        "train": dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding="max_length",
        )

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    print("--- Dataset loaded and prepared successfully ---")
    return tokenized_dataset
