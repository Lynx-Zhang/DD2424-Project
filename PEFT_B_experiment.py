#!/usr/bin/env python3

"""
PEFT B-grade experiment runner.
Compares full, last-linear, LoRA, and ReFT on SST dev.
"""

import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from peft_classifier import (
    GPT2SentimentClassifier,
    SentimentDataset,
    load_data,
    model_eval,
    seed_everything,
)
from peft_utils import count_trainable_parameters
from optimizer import AdamW


# Paths and runtime configuration
TRAIN_PATH = "data/ids-sst-train.csv"
DEV_PATH = "data/ids-sst-dev.csv"
USE_GPU = True
DEVICE = torch.device("cuda") if USE_GPU and torch.cuda.is_available() else torch.device("cpu")

# Training hyperparameters
EPOCHS = 3
BATCH_SIZE = 8
LR = 1e-3
HIDDEN_DROPOUT_PROB = 0.1

# PEFT hyperparameters
LORA_R = 8
LORA_ALPHA = 16
REFT_R = 4
REFT_LAYERS = 2
REFT_POSITIONS = "last"

SEEDS = [11711, 11712, 11713]
METHODS = [
    {"name": "full", "peft_type": "full"},
    {"name": "last-linear", "peft_type": "none"},
    {"name": "lora", "peft_type": "lora"},
    {"name": "reft", "peft_type": "reft"},
]


def run_single_experiment(method, seed):
    seed_everything(seed)

    train_data, num_labels = load_data(TRAIN_PATH, "train")
    dev_data = load_data(DEV_PATH, "valid")

    train_dataset = SentimentDataset(train_data, SimpleNamespace())
    dev_dataset = SentimentDataset(dev_data, SimpleNamespace())

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=train_dataset.collate_fn,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        collate_fn=dev_dataset.collate_fn,
    )

    config = SimpleNamespace(
        peft_type=method["peft_type"],
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        reft_r=REFT_R,
        reft_layers=REFT_LAYERS,
        reft_positions=REFT_POSITIONS,
        hidden_dropout_prob=HIDDEN_DROPOUT_PROB,
        num_labels=num_labels,
        hidden_size=768,
    )

    model = GPT2SentimentClassifier(config).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)

    start_time = time.time()
    model.train()
    for _ in range(EPOCHS):
        for batch in train_dataloader:
            b_ids = batch["token_ids"].to(DEVICE)
            b_mask = batch["attention_mask"].to(DEVICE)
            b_labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction="sum") / BATCH_SIZE
            loss.backward()
            optimizer.step()
    runtime = time.time() - start_time

    dev_acc, dev_f1 = model_eval(dev_dataloader, model, DEVICE)
    trainable_params = count_trainable_parameters(model)

    return {
        "method": method["name"],
        "seed": seed,
        "dev_acc": float(dev_acc),
        "dev_f1": float(dev_f1),
        "trainable_params": int(trainable_params),
        "runtime_sec": float(runtime),
    }


def main():
    import pandas as pd
    import matplotlib.pyplot as plt

    all_results = []
    for method in METHODS:
        for seed in SEEDS:
            result = run_single_experiment(method, seed)
            all_results.append(result)
            print(result)

    df = pd.DataFrame(all_results)
    print("\nRaw results:")
    print(df)
    df.to_csv("peft_b_results_raw.csv", index=False)

    summary = (
        df.groupby("method")
        .agg(
            {
                "dev_acc": ["mean", "std"],
                "dev_f1": ["mean", "std"],
                "trainable_params": ["mean"],
                "runtime_sec": ["mean"],
            }
        )
        .reset_index()
    )
    print("\nSummary:")
    print(summary)
    summary.to_csv("peft_b_results_summary.csv", index=False)

    methods = summary["method"].tolist()
    acc_mean = summary[("dev_acc", "mean")].tolist()
    acc_std = summary[("dev_acc", "std")].fillna(0.0).tolist()

    plt.figure(figsize=(7, 4))
    plt.bar(methods, acc_mean, yerr=acc_std, capsize=4, color="#4c72b0")
    plt.title("PEFT Comparison: Dev Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Method")
    plt.tight_layout()
    plt.savefig("peft_b_dev_acc.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
