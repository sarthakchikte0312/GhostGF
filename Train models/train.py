"""
train.py
QLoRA fine-tune a small open-weight chat model on your training_data.jsonl
(produced by data_prep.py) using Unsloth - fast and memory-efficient enough
to run on a single consumer NVIDIA GPU (8GB+ VRAM).

Install (check unsloth's GitHub for the exact command matching your CUDA
version - this is the general pattern):
    pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
    pip install trl datasets

Usage:
    python train.py --data training_data.jsonl --out lora_adapter
"""

import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# Ungated models that work well and don't need HuggingFace approval.
# Pick based on your VRAM (check with `nvidia-smi`):
#   ~8GB VRAM   -> "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  (default below)
#   ~12GB+ VRAM -> same model, just bump per_device_train_batch_size
#   ~24GB+ VRAM -> "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" (needs HF gated access)
DEFAULT_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LEN = 2048


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="training_data.jsonl from data_prep.py")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="lora_adapter")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    dataset = load_dataset("json", data_files=args.data, split="train")

    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(format_example)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            logging_steps=10,
            output_dir=args.out,
            save_strategy="epoch",
            optim="adamw_8bit",
        ),
    )

    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
