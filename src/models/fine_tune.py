import os
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftConfig, PeftModel


PROJECT_DIR_PATH = ''
OUTPUT_PATH = os.path.join(PROJECT_DIR_PATH, 'models')

BASELINE_MODEL_NAME = 'HuggingFaceH4/zephyr-7b-beta'
USER_NAME = "jakemannix"
PROJECT_NAME = 'zephyr-7b-beta_assistant_v0.2'
HUGGING_FACE_REPO_NAME = f'{USER_NAME}/{PROJECT_NAME}'
HUGGING_FACE_MERGED_REPO_NAME = f'{HUGGING_FACE_REPO_NAME}_merged'
HUGGING_FACE_GPTQ_REPO_NAME = f'{HUGGING_FACE_REPO_NAME}_gptq'

# Set Hyperparameters.
MAXLEN = 512
BATCH_SIZE = 6
GRAD_ACC = 4
WARMUP = 100
STEPS = 1000
OPTIMIZER = 'paged_adamw_8bit'  # Use paged optimizer to save memory
LR = 4e-5                       # Use value slightly smaller than pretraining lr value & close to LoRA standard


def setup():
    try:
        if 'WANDB_API_KEY' not in os.environ:
            from getpass import getpass
            os.environ['WANDB_API_KEY'] = getpass('Enter your wandb API key: ')
        import wandb
        wandb.login()
        wandb.init(project=PROJECT_NAME)
        import huggingface_hub
        huggingface_hub.login()
    except ImportError as e:
        print(f"ImportError: {e}")
        print("WandB and/or Huggingface Hub not installed.")


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME,
                                              padding_side='left',
                                              add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_base_model():
    model = AutoModelForCausalLM.from_pretrained(BASELINE_MODEL_NAME,
                                                 load_in_8bit=True,
                                                 device_map='auto')
    return model


def get_training_config():
    # Set training config.
    training_config = transformers.TrainingArguments(per_device_train_batch_size=BATCH_SIZE,
                                                     gradient_accumulation_steps=GRAD_ACC,
                                                     warmup_steps=WARMUP,
                                                     max_steps=STEPS,
                                                     optim=OPTIMIZER,
                                                     learning_rate=LR,
                                                     # fp16=True,  # Consider using bf16 if compatible with your GPU
                                                     logging_steps=1,
                                                     output_dir=OUTPUT_PATH,
                                                     report_to=['wandb'],
                                                     load_best_model_at_end=True,
                                                     evaluation_strategy='steps',
                                                     metric_for_best_model='eval_loss',
                                                     greater_is_better=False,
                                                     eval_steps=10,
                                                     save_steps=10,
                                                     save_total_limit=2)

    return training_config


def get_trainer(model, train_data, val_data, tokenizer):
    training_config = get_training_config()
    # Stabilize output layer and layernorms & prepare for 8bit training.
    model = prepare_model_for_kbit_training(model, 8)
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Apply to "q_proj", "v_proj" layers of attention as suggested by paper
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM")
    # Set PEFT adapter on model.
    model = get_peft_model(model, config)
    model.config.use_cache = False  # Silence the warnings.
    # Setup collator.
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    early_stop = transformers.EarlyStoppingCallback(10, 1.15)
    # Setup trainer.
    trainer = transformers.Trainer(model=model,
                                   train_dataset=train_data,
                                   eval_dataset=val_data,
                                   data_collator=data_collator,
                                   args=training_config,
                                   callbacks=[early_stop])
    return trainer


def push_to_hub(model, tokenizer):
    # Push model to Huggingface Hub.
    # TODO: commit message?
    model.push_to_hub(HUGGING_FACE_REPO_NAME, use_auth_token=True)
    tokenizer.push_to_hub(HUGGING_FACE_REPO_NAME, use_auth_token=True)


def merge_peft_model_and_push_to_hub(tokenizer):
    config = PeftConfig.from_pretrained(HUGGING_FACE_REPO_NAME)
    # Get base model
    model = transformers.AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                                              torch_dtype=torch.float16,
                                                              # GPTQ quantization requires fp16
                                                              return_dict=True)
    model = PeftModel.from_pretrained(model, HUGGING_FACE_REPO_NAME)
    # Merge model and Lora adapter.
    merged_model = model.merge_and_unload()
    merged_model.push_to_hub(HUGGING_FACE_MERGED_REPO_NAME)
    tokenizer.push_to_hub(HUGGING_FACE_MERGED_REPO_NAME)
    return merged_model, tokenizer

