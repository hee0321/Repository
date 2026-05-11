from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# --------------------------------------------------------------------------------
# 1. Model & Tokenizer Setup
# --------------------------------------------------------------------------------
model_name = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Instruct" # Base model
max_seq_length = 4096 # Large context for long CoT reasoning

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # Highly efficient 4-bit loading
)

# --------------------------------------------------------------------------------
# 2. Add LoRA Adapters
# --------------------------------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Max rank for high complexity reasoning
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# --------------------------------------------------------------------------------
# 3. Dataset Preparation
# --------------------------------------------------------------------------------
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    responses    = examples["response"]
    texts = []
    for instruction, response in zip(instructions, responses):
        # Using the standard ChatML / Instruct format for Nemotron
        text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("json", data_files="synthetic_nemotron_train.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# --------------------------------------------------------------------------------
# 4. Training Arguments
# --------------------------------------------------------------------------------
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 100,
        max_steps = 1000, # Initial run. Increase for 0.9 goal.
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_available(),
        bf16 = torch.cuda.is_available(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# --------------------------------------------------------------------------------
# 5. Execute Training
# --------------------------------------------------------------------------------
print("Starting Nemotron Fine-tuning...")
trainer_stats = trainer.train()

# --------------------------------------------------------------------------------
# 6. Save LoRA Adapter
# --------------------------------------------------------------------------------
model.save_pretrained_merged("nemotron_lora_adapter", tokenizer, save_method = "lora")
print("Training complete. Adapter saved in 'nemotron_lora_adapter' directory.")
