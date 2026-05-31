#!/usr/bin/env python3

"""
Parse raw log text into CSV and summary tables.
"""

import argparse
import ast
import csv
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="rst.txt")
    parser.add_argument("--output_raw", default="peft_partial_raw.csv")
    parser.add_argument("--output_summary", default="peft_partial_summary.csv")
    return parser.parse_args()


def extract_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                record = ast.literal_eval(line)
            except Exception:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def write_raw(records, path):
    if not records:
        return
    fieldnames = sorted({k for r in records for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def write_summary(records, path):
    if not records:
        return
    groups = defaultdict(list)
    for r in records:
        key = (
            r.get("peft_type"),
            tuple(r.get("lora_targets") or []),
            r.get("lora_r"),
            r.get("lora_alpha"),
            r.get("reft_r"),
            r.get("reft_layers"),
            r.get("reft_positions"),
        )
        groups[key].append(r)

    rows = []
    for key, vals in groups.items():
        accs = [v.get("dev_acc") for v in vals if v.get("dev_acc") is not None]
        f1s = [v.get("dev_f1") for v in vals if v.get("dev_f1") is not None]
        if not accs:
            continue
        acc_mean = sum(accs) / len(accs)
        f1_mean = sum(f1s) / len(f1s) if f1s else None
        rows.append(
            {
                "peft_type": key[0],
                "lora_targets": list(key[1]),
                "lora_r": key[2],
                "lora_alpha": key[3],
                "reft_r": key[4],
                "reft_layers": key[5],
                "reft_positions": key[6],
                "count": len(vals),
                "dev_acc_mean": acc_mean,
                "dev_f1_mean": f1_mean,
            }
        )

    fieldnames = [
        "peft_type",
        "lora_targets",
        "lora_r",
        "lora_alpha",
        "reft_r",
        "reft_layers",
        "reft_positions",
        "count",
        "dev_acc_mean",
        "dev_f1_mean",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    args = parse_args()
    records = extract_records(args.input)
    write_raw(records, args.output_raw)
    write_summary(records, args.output_summary)
    print(f"Parsed {len(records)} records")


if __name__ == "__main__":
    main()
