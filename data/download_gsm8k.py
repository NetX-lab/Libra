"""Support code for Download gsm8k."""

from datasets import load_dataset
import json
import os

save_dir = os.path.dirname(os.path.abspath(__file__))

print("Downloading the GSM8K dataset...")
dataset = load_dataset("openai/gsm8k", "main")


train_path = os.path.join(save_dir, "gsm8k_train.jsonl")
dataset["train"].to_json(train_path)
print(f"Saved train split: {train_path} ({len(dataset['train'])} samples)")


test_path = os.path.join(save_dir, "gsm8k_test.jsonl")
dataset["test"].to_json(test_path)
print(f"Saved test split: {test_path} ({len(dataset['test'])} samples)")

print("\nDone. Transfer these files to the target machine:")
print(f"  - {train_path}")
print(f"  - {test_path}")
