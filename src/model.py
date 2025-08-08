from unsloth import FastLanguageModel
import torch

def load_model_and_tokenizer(model_config):
    """
    Loads the model and tokenizer from Unsloth.

    Args:
        model_config (dict): A dictionary containing model configuration.
                             Expected keys: 'name', 'max_seq_length', 'dtype', 'load_in_8bit'.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    print(f"--- Loading model: {model_config['name']} ---")

    # Determine the torch data type
    if model_config['dtype'] == 'bfloat16':
        dtype = torch.bfloat16
    elif model_config['dtype'] == 'float16':
        dtype = torch.float16
    else:
        dtype = None

    kwargs = {
        'model_name': model_config['name'],
        'max_seq_length': model_config['max_seq_length'],
        'dtype': dtype,
    }
    if 'load_in_8bit' in model_config:
        kwargs['load_in_8bit'] = bool(model_config['load_in_8bit'])

    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)

    print("--- Model and tokenizer loaded successfully ---")
    return model, tokenizer

def apply_peft(model, lora_config):
    """
    Applies PEFT LoRA configuration to the model.

    Args:
        model: The model to which PEFT will be applied.
        lora_config (dict): A dictionary containing LoRA configuration.

    Returns:
        The model with the PEFT configuration.
    """
    print("--- Applying PEFT LoRA configuration ---")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config['r'],
        target_modules=lora_config['target_modules'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        bias=lora_config['bias'],
        use_gradient_checkpointing=True,
        random_state=42,
        max_seq_length=model.max_seq_length,
    )
    print("--- PEFT applied successfully ---")
    return model
