"""Support code for Test single gpu."""

import sys
import os


_current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_current_dir))

import torch
from RL_Framework.engine.train_engine import FSDPTrainEngine
from RL_Framework.config import AsyncRLConfig

print("=" * 60)
print("Single-GPU training-engine initialization test")
print("=" * 60)


config = AsyncRLConfig(
    model_path="Qwen/Qwen2-1.5B-Instruct",
    train_gpus=1,
    rollout_gpus=0,
)

print("\nConfiguration:")
print(f"  model_path: {config.model_path}")
print(f"  train_gpus: {config.train_gpus}")


print("\nCreating the training engine...")
train_engine = FSDPTrainEngine(
    model_path=config.model_path,
    learning_rate=config.learning_rate,
    clip_epsilon=config.clip_epsilon,
    kl_coef=config.kl_coef,
)


print("\nInitializing the model...")
try:
    train_engine.initialize(max_seq_length=config.max_seq_length)
    print("\nModel initialized successfully.")
    print(f"  is_distributed: {train_engine.is_distributed}")
    print(f"  is_main_process: {train_engine.is_main_process}")
    print(f"  device: {train_engine.local_rank}")


    print("\nTesting weight saving...")
    test_path = "/path/to/user/test_weights"
    train_engine.save_weights(test_path, version=0)
    print("Weights saved successfully.")


    print("\nTesting weight loading...")
    train_engine.load_weights(test_path, version=0)
    print("Weights loaded successfully.")

    print("\n" + "=" * 60)
    print("All tests passed; single-GPU mode is working.")
    print("=" * 60)

except Exception as e:
    print(f"\nTest failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
