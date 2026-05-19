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
    parser.add_argument("--mode", choices=["lora", "reft", "both"], default="both")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--max_minutes", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
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


def config_key(cfg):
    return (
        cfg.peft_type,
        tuple(cfg.lora_targets) if cfg.lora_targets else (),
        cfg.lora_r,
        cfg.lora_alpha,
        cfg.reft_r,
        cfg.reft_layers,
        cfg.reft_positions,
        cfg.seed,
        cfg.lr,
        cfg.epochs,
        cfg.batch_size,
    )


def load_completed(output_path):
    completed = set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                completed.add(
                    (
                        record.get("peft_type"),
                        tuple(record.get("lora_targets") or []),
                        record.get("lora_r"),
                        record.get("lora_alpha"),
                        record.get("reft_r"),
                        record.get("reft_layers"),
                        record.get("reft_positions"),
                        record.get("seed"),
                        record.get("lr"),
                        record.get("epochs"),
                        record.get("batch_size"),
                    )
                )
    except FileNotFoundError:
        return completed
    return completed


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

    all_jobs = []
    for seed in args.seeds:
        if args.mode in ["lora", "both"]:
            for lora_r, lora_alpha, lora_targets in lora_configs:
                all_jobs.append(
                    ("lora", seed, (lora_r, lora_alpha, lora_targets))
                )
        if args.mode in ["reft", "both"]:
            for reft_r, reft_layers, reft_positions in reft_configs:
                all_jobs.append(
                    ("reft", seed, (reft_r, reft_layers, reft_positions))
                )

    start_idx = max(args.start_index, 0)
    end_idx = len(all_jobs) if args.end_index < 0 else min(args.end_index, len(all_jobs))
    jobs = all_jobs[start_idx:end_idx]
    if args.chunk_size > 0:
        jobs = jobs[: args.chunk_size]

    completed = load_completed(args.output) if args.resume else set()
    start_time = time.time()
    total_jobs = len(jobs)
    done_jobs = 0

    with open(args.output, "a", encoding="utf-8") as out:
        for peft_type, seed, params in jobs:
            if args.max_minutes > 0 and (time.time() - start_time) > args.max_minutes * 60:
                print("Reached time limit, stopping early.")
                break

            seed_everything(seed)
            train_dataloader, dev_dataloader, num_labels = build_dataloaders(
                args.train, args.dev, args.batch_size
            )

            if peft_type == "lora":
                lora_r, lora_alpha, lora_targets = params
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
            else:
                reft_r, reft_layers, reft_positions = params
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

            key = config_key(cfg)
            if key in completed:
                done_jobs += 1
                print(f"Progress: {done_jobs}/{total_jobs} (skipped)")
                continue

            result = run_one(cfg, train_dataloader, dev_dataloader, device)
            out.write(json.dumps(result) + "\n")
            out.flush()
            done_jobs += 1
            print(f"Progress: {done_jobs}/{total_jobs}")
            print(result)


if __name__ == "__main__":
    main()
