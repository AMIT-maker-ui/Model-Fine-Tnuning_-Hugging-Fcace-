"""
train.py — Fine-tune DialoGPT-small for Conversational AI
==========================================================
Model   : microsoft/DialoGPT-small  (low RAM, fast, laptop-friendly)
Dataset : Custom CSV or DailyDialog (auto-downloaded)
Task    : Conversational response generation
Output  : Fine-tuned model saved locally + pushed to Hugging Face Hub

Run:
    pip install -r requirements.txt
    python train.py
"""

import os
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from huggingface_hub import login

HF_TOKEN    = "HF_TOKENS"
HF_REPO_ID  = "hari-krishna-ai/my-chatbot"   
BASE_MODEL  = "microsoft/DialoGPT-small"  
OUTPUT_DIR  = "./chatbot-output"
MAX_LENGTH  = 128     
EPOCHS      = 3       
BATCH_SIZE  = 4       
SAVE_STEPS  = 100
LOG_STEPS   = 10



def load_daily_dialog_subset(num_samples: int = 1000):
    """
    Load DailyDialog dataset from Hugging Face.
    Falls back to a tiny built-in demo set if download fails.
    Returns a list of (input, response) string pairs.
    """
    try:
        print("📦 Downloading DailyDialog dataset (small subset)...")
        raw = load_dataset("daily_dialog", split=f"train[:{num_samples}]", trust_remote_code=True)
        pairs = []
        for item in raw:
            dialog = item["dialog"]
            for i in range(len(dialog) - 1):
                user_msg = dialog[i].strip()
                bot_msg  = dialog[i + 1].strip()
                if user_msg and bot_msg:
                    pairs.append({"input": user_msg, "response": bot_msg})
        print(f"   ✅ Loaded {len(pairs)} conversation pairs from DailyDialog")
        return pairs
    except Exception as e:
        print(f"   ⚠️  DailyDialog download failed ({e}). Using built-in demo data.")
        return _builtin_demo_data()


def _builtin_demo_data():
    """Minimal demo data so training always works offline."""
    return [
        {"input": "Hello!",                       "response": "Hi there! How are you doing today?"},
        {"input": "How are you?",                  "response": "I'm doing great, thanks for asking! What about you?"},
        {"input": "What's your name?",             "response": "I'm an AI assistant. You can call me Aria!"},
        {"input": "What can you do?",              "response": "I can chat with you, answer questions, and help with various topics."},
        {"input": "Tell me a joke.",               "response": "Why don't scientists trust atoms? Because they make up everything!"},
        {"input": "What's the weather like?",      "response": "I don't have access to real-time data, but I hope it's sunny where you are!"},
        {"input": "Good morning!",                 "response": "Good morning! Hope you have a wonderful day ahead!"},
        {"input": "I'm feeling sad today.",        "response": "I'm sorry to hear that. Want to talk about what's bothering you?"},
        {"input": "What's 2 + 2?",                "response": "That's 4! Simple math is my specialty too."},
        {"input": "Recommend a book.",             "response": "I'd suggest 'Atomic Habits' by James Clear. It's life-changing!"},
        {"input": "Do you like music?",            "response": "I love discussing music! What genre do you enjoy?"},
        {"input": "Goodbye!",                      "response": "Goodbye! It was great chatting with you. Come back anytime!"},
    ]


def build_hf_dataset(pairs: list, tokenizer) -> Dataset:
    """
    Convert conversation pairs into tokenized Hugging Face Dataset.
    Format: '<|user|> {input} <|bot|> {response} <|endoftext|>'
    """
    texts = []
    for p in pairs:
        text = f"<|user|> {p['input']} <|bot|> {p['response']} {tokenizer.eos_token}"
        texts.append(text)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    raw_ds = Dataset.from_dict({"text": texts})
    tokenized = raw_ds.map(tokenize, batched=True, remove_columns=["text"])
    tokenized = tokenized.map(lambda x: {"labels": x["input_ids"].copy()})
    tokenized.set_format("torch")
    return tokenized


def main():
    # ── Step 1: Authenticate ──────────────────────────────────
    print("\n🔐 Logging in to Hugging Face Hub...")
    login(HF_TOKEN)

    # ── Step 2: Load tokenizer & model ───────────────────────
    print(f"\n🧠 Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token   # DialoGPT has no pad token by default

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.resize_token_embeddings(len(tokenizer))

    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {'GPU ✅' if torch.cuda.is_available() else 'CPU (training may be slow)'}")

    # ── Step 3: Prepare dataset ───────────────────────────────
    pairs   = load_daily_dialog_subset(num_samples=2000)
    split   = int(len(pairs) * 0.9)
    train_d = build_hf_dataset(pairs[:split], tokenizer)
    eval_d  = build_hf_dataset(pairs[split:], tokenizer)
    print(f"\n📊 Dataset ready — Train: {len(train_d)} | Eval: {len(eval_d)}")

    # ── Step 4: Training arguments ────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),     # Auto-enable mixed precision on GPU
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",                   # Disable wandb/tensorboard
        push_to_hub=False,                  # We push manually after training
        hub_model_id=HF_REPO_ID,
    )

    # ── Step 5: Trainer ───────────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_d,
        eval_dataset=eval_d,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # ── Step 6: Train ─────────────────────────────────────────
    print("\n🚀 Starting training...\n")
    trainer.train()
    print("\n✅ Training complete!")

    # ── Step 7: Save locally ──────────────────────────────────
    print(f"\n💾 Saving model to '{OUTPUT_DIR}'...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # ── Step 8: Push to Hugging Face Hub ──────────────────────
    print(f"\n☁️  Pushing to Hugging Face Hub: {HF_REPO_ID}")
    trainer.push_to_hub(commit_message="🤖 Fine-tuned DialoGPT chatbot")
    tokenizer.push_to_hub(HF_REPO_ID)
    print(f"\n🎉 Model live at: https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
