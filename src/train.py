from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState
import os
import torch
import math

class PerplexityLoggingCallback(TrainerCallback):
    """
    A custom callback to compute and log perplexity, preventing memory leaks.
    """
    def on_log(self, args: TrainingArguments, state: TrainerState, control=None, logs=None, **kwargs):
        """
        Event called after logging. We access the state to prevent memory leaks.
        """
        # Get the last log entry, which was just added by the Trainer.
        # This is where the tensor is stored, causing the memory leak.
        if state.log_history:
            last_log = state.log_history[-1]

            if 'loss' in last_log:
                loss_value = last_log['loss']
                if hasattr(loss_value, 'item'):  # Check if it's a tensor
                    detached_loss = loss_value.item()
                else:
                    detached_loss = float(loss_value)
                
                # Overwrite the tensor in the history with a detached float.
                # THIS IS THE CRITICAL FIX.
                state.log_history[-1]['loss'] = detached_loss

                try:
                    perplexity = math.exp(detached_loss)
                    # Also add perplexity to the history log for consistency.
                    state.log_history[-1]['perplexity'] = perplexity
                except (OverflowError, ValueError):
                    state.log_history[-1]['perplexity'] = float('inf')
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control=None, **kwargs):
        """
        Event called at the end of a training step.
        """
        # Log GPU memory usage at each logging step to track baseline memory.
        if state.global_step > 0 and state.global_step % args.logging_steps == 0:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"--- Step {state.global_step}: GPU mem allocated: {allocated:.2f} GiB, reserved: {reserved:.2f} GiB ---")


def train_model(config, model, tokenizer, train_dataset, eval_dataset):
    """
    Sets up and runs the training process.
    """
    print("--- Setting up training ---")

    # Ensure W&B project is applied if provided in config
    try:
        wandb_project = config.get('env', {}).get('wandb_project')
        if wandb_project:
            os.environ.setdefault("WANDB_PROJECT", str(wandb_project))
    except Exception:
        pass

    # Handle different versions of the TrainingArguments API
    training_args_dict = config['training'].copy()
    if 'evaluation_strategy' in training_args_dict:
        training_args_dict['eval_strategy'] = training_args_dict.pop('evaluation_strategy')


    training_args = TrainingArguments(
        output_dir=config['paths']['output_dir'],
        per_device_train_batch_size=training_args_dict['per_device_train_batch_size'],
        gradient_accumulation_steps=training_args_dict['gradient_accumulation_steps'],
        warmup_steps=training_args_dict['warmup_steps'],
        max_steps=training_args_dict['max_steps'],
        learning_rate=training_args_dict['learning_rate'],
        lr_scheduler_type=training_args_dict['lr_scheduler_type'],
        fp16=not training_args_dict['bf16'],
        bf16=training_args_dict['bf16'],
        logging_steps=training_args_dict['logging_steps'],
        eval_strategy=training_args_dict.get('eval_strategy'),
        eval_steps=training_args_dict['eval_steps'],
        save_steps=training_args_dict['save_steps'],
        optim=training_args_dict['optimizer'],
        group_by_length=True,
        gradient_checkpointing=True,
        weight_decay=0.01,
        seed=42,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        run_name=f"finetune-{config['model']['name']}",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[PerplexityLoggingCallback()],
    )

    print("--- Starting training ---")
    trainer.train()
    print("--- Training finished ---")

    # Save the final adapter
    final_adapter_path = os.path.join(config['paths']['adapter_dir'], "final_adapter")
    model.save_pretrained(final_adapter_path)
    print(f"Final adapter saved to {final_adapter_path}")
