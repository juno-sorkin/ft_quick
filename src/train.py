from transformers import Trainer, TrainingArguments
import os

class PerplexityTrainer(Trainer):
    """
    A custom trainer to compute and log perplexity.
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss and calculates perplexity.
        """
        outputs = model(**inputs)
        loss = outputs.loss

        # Calculate perplexity
        try:
            perplexity = torch.exp(loss)
        except:
            # Handle cases where loss is not a tensor or other issues
            perplexity = float('inf')

        self.log({"perplexity": perplexity.item()})

        return (loss, outputs) if return_outputs else loss


def train_model(config, model, tokenizer, train_dataset, eval_dataset):
    """
    Sets up and runs the training process.

    Args:
        config (dict): The main configuration dictionary.
        model: The model to be trained.
        tokenizer: The tokenizer associated with the model.
        train_dataset: The training dataset.
        eval_dataset: The evaluation dataset.
    """
    print("--- Setting up training ---")

    training_args = TrainingArguments(
        output_dir=config['paths']['output_dir'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        warmup_steps=config['training']['warmup_steps'],
        max_steps=config['training']['max_steps'],
        learning_rate=config['training']['learning_rate'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        fp16=not config['training']['bf16'],
        bf16=config['training']['bf16'],
        logging_steps=config['training']['logging_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        optim=config['training']['optimizer'],
        weight_decay=0.01,
        seed=42,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        run_name=f"finetune-{config['model']['name']}",
    )

    trainer = PerplexityTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # Data collator can be added here if needed
    )

    print("--- Starting training ---")
    trainer.train()
    print("--- Training finished ---")

    # Save the final adapter
    final_adapter_path = os.path.join(config['paths']['adapter_dir'], "final_adapter")
    model.save_pretrained(final_adapter_path)
    print(f"Final adapter saved to {final_adapter_path}")
