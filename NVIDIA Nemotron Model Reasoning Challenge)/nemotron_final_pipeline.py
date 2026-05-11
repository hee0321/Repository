import os
import json
import random
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# --------------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------------
NUM_SAMPLES = 314 
# Using the official Unsloth Nemotron-8B model
MODEL_NAME = "unsloth/Llama-3.1-Nemotron-Nano-8B-v1" 
MAX_SEQ_LENGTH = 1024
OUTPUT_DIR = "nemotron_lora_adapter"

# --------------------------------------------------------------------------------
# 2. Logic
# --------------------------------------------------------------------------------

def generate_reasoning_sample():
    # Complex logical deduction samples
    ops = ["XOR", "NOR", "NAND", "NXOR"]
    op = random.choice(ops)
    a, b = random.randint(0, 15), random.randint(0, 15)
    
    if op == "XOR": res = a ^ b
    elif op == "NOR": res = ~(a | b) & 0x0F
    elif op == "NAND": res = ~(a & b) & 0x0F
    else: res = ~(a ^ b) & 0x0F # NXOR
    
    instruction = f"Calculate the 4-bit result of {a} {op} {b}."
    response = f"Thought: {a} is {bin(a)[2:].zfill(4)}, {b} is {bin(b)[2:].zfill(4)}.\nApplying {op} results in {bin(res)[2:].zfill(4)}.\nFinal Answer: \\boxed{{{res}}}"
    return {"instruction": instruction, "response": response}

def main():
    print(f"Generating {NUM_SAMPLES} samples...")
    dataset_list = [generate_reasoning_sample() for _ in range(NUM_SAMPLES)]
    dataset = Dataset.from_list(dataset_list)
    
    def format_fn(examples):
        texts = []
        for i, r in zip(examples["instruction"], examples["response"]):
            texts.append(f"<|im_start|>user\n{i}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>")
        return {"text": texts}
    dataset = dataset.map(format_fn, batched=True)

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True, # Enable 4-bit loading on-the-fly
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 32, # Higher rank for better reasoning
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    print("Starting Training (Fast Mode)...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 20, 
            learning_rate = 1e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            output_dir = "outputs",
            remove_unused_columns = False,
        ),
    )

    trainer.train()

    print("Saving and Zipping...")
    # Important: Save in a way that Kaggle can load
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    import shutil
    shutil.make_archive("submission", 'zip', OUTPUT_DIR)
    print("SUCCESS! submission.zip is ready for upload.")

if __name__ == "__main__":
    main()
