import argparse
import yaml
from pprint import pprint

from model import load_model_and_tokenizer, apply_peft
from data_loader import load_and_prepare_dataset
from train import train_model

def main():
    """
    Main function to run the fine-tuning pipeline.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a language model.")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test", "inference"],
        help="The operational mode.",
    )
    args = parser.parse_args()

    # Load configuration
    with open("config/config.yml", "r") as f:
        config = yaml.safe_load(f)

    print("--- Configuration ---")
    pprint(config)
    print("---------------------")

    if args.mode == "train":
        print("--- Starting Training Mode ---")
        # 1. Load tokenizer and model
        model, tokenizer = load_model_and_tokenizer(config['model'])
        model = apply_peft(model, config['lora'])

        # 2. Load and prepare dataset
        dataset = load_and_prepare_dataset(config['data'], tokenizer)

        # 3. Start training
        train_model(config, model, tokenizer, dataset['train'], dataset['validation'])

    elif args.mode == "test":
        print("--- Starting Test Mode ---")
        # TODO: implement testing logic
        print("Placeholder for testing logic.")

    elif args.mode == "inference":
        print("--- Starting Inference Mode ---")
        # TODO: implement inference logic
        print("Placeholder for inference logic.")

    else:
        raise ValueError(f"Invalid mode: {args.mode}")

# TODO: add inference logic

if __name__ == "__main__":
    main()
