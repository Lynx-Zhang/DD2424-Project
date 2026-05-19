#!/usr/bin/env python3

"""
PEFT grid + ablation runner.
Scans LoRA/ReFT hyperparameters and structure/position ablations.
"""

import argparse
import itertools
import json
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/ids-sst-train.csv")
    parser.add_argument("--dev", default="data/ids-sst-dev.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11711, 11712, 11713])
    parser.add_argument("--output", default="peft_grid_results.jsonl")
    return parser.parse_args()


def build_dataloaders(train_path, dev_path, batch_size):
    train_data, num_labels = load_data(train_path, "train")
    dev_data = load_data(dev_path, "valid")

    train_dataset = SentimentDataset(train_data, SimpleNamespace())
    dev_dataset = SentimentDataset(dev_data, SimpleNamespace())

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=dev_dataset.collate_fn,
    )
    return train_dataloader, dev_dataloader, num_labels


def run_one(cfg, train_dataloader, dev_dataloader, device):
    model = GPT2SentimentClassifier(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    start_time = time.time()
    model.train()
    for _ in range(cfg.epochs):
        for batch in train_dataloader:
            b_ids = batch["token_ids"].to(device)
            b_mask = batch["attention_mask"].to(device)
            b_labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction="sum") / cfg.batch_size
            loss.backward()
            optimizer.step()
    runtime = time.time() - start_time

    dev_acc, dev_f1 = model_eval(dev_dataloader, model, device)
    trainable_params = count_trainable_parameters(model)

    return {
        "peft_type": cfg.peft_type,
        "lora_targets": cfg.lora_targets,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "reft_r": cfg.reft_r,
        "reft_layers": cfg.reft_layers,
        "reft_positions": cfg.reft_positions,
        "dev_acc": float(dev_acc),
        "dev_f1": float(dev_f1),
        "trainable_params": int(trainable_params),
        "runtime_sec": float(runtime),
        "seed": cfg.seed,
    }


def main():
    args = parse_args()
    device = torch.device("cuda") if args.use_gpu and torch.cuda.is_available() else torch.device("cpu")

    # Parameter grids
    lora_r_grid = [2, 4, 8, 16]
    lora_alpha_grid = [8, 16, 32]
    lora_targets_grid = [
        ["query"],
        ["query", "value"],
        ["query", "key", "value"],
    ]

    reft_r_grid = [2, 4, 8]
    reft_layers_grid = [1, 2, 4]
    reft_positions_grid = ["first", "last", "all"]

    lora_configs = list(itertools.product(lora_r_grid, lora_alpha_grid, lora_targets_grid))
    reft_configs = list(itertools.product(reft_r_grid, reft_layers_grid, reft_positions_grid))

    with open(args.output, "w", encoding="utf-8") as out:
        for seed in args.seeds:
            seed_everything(seed)

            train_dataloader, dev_dataloader, num_labels = build_dataloaders(
                args.train, args.dev, args.batch_size
            )

            # LoRA scans + target ablation
            for lora_r, lora_alpha, lora_targets in lora_configs:
                cfg = SimpleNamespace(
                    peft_type="lora",
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_targets=lora_targets,
                    reft_r=4,
                    reft_layers=2,
                    reft_positions="last",
                    hidden_dropout_prob=0.1,
                    num_labels=num_labels,
                    hidden_size=768,
                    lr=args.lr,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    seed=seed,
                )
                result = run_one(cfg, train_dataloader, dev_dataloader, device)
                out.write(json.dumps(result) + "\n")
                out.flush()
                print(result)

            # ReFT scans + position ablation
            for reft_r, reft_layers, reft_positions in reft_configs:
                cfg = SimpleNamespace(
                    peft_type="reft",
                    lora_r=8,
                    lora_alpha=16,
                    lora_targets=["query", "key", "value"],
                    reft_r=reft_r,
                    reft_layers=reft_layers,
                    reft_positions=reft_positions,
                    hidden_dropout_prob=0.1,
                    num_labels=num_labels,
                    hidden_size=768,
                    lr=args.lr,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    seed=seed,
                )
                result = run_one(cfg, train_dataloader, dev_dataloader, device)
                out.write(json.dumps(result) + "\n")
                out.flush()
                print(result)


if __name__ == "__main__":
    main()
