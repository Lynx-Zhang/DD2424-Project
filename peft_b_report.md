# PEFT B-Grade Experiment Report

## Overview
We compare four finetuning strategies on SST dev:
- Full finetuning (all parameters trainable)
- Last-linear (only the classifier head)
- LoRA (low-rank adapters on Q/K/V)
- ReFT (low-rank representation intervention)

Each method is evaluated with 3 random seeds. We report dev accuracy, macro F1, trainable parameter count, and runtime.

## Experimental Setup
- Dataset: SST (train/dev)
- Seeds: 11711, 11712, 11713
- Epochs: 3
- Batch size: 8
- Learning rate: 1e-3
- LoRA: r=8, alpha=16
- ReFT: r=4, last 2 layers, positions=last

## Raw Results (per seed)
| Method | Seed | Dev Acc | Dev F1 | Trainable Params | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| full | 11711 | 0.3851 | 0.3122 | 125,034,245 | 400.66 |
| full | 11712 | 0.3878 | 0.3801 | 125,034,245 | 403.16 |
| full | 11713 | 0.3960 | 0.3858 | 125,034,245 | 403.98 |
| last-linear | 11711 | 0.3960 | 0.3420 | 3,845 | 73.98 |
| last-linear | 11712 | 0.4196 | 0.4142 | 3,845 | 73.99 |
| last-linear | 11713 | 0.4405 | 0.3176 | 3,845 | 73.75 |
| lora | 11711 | 0.4523 | 0.4172 | 446,213 | 197.13 |
| lora | 11712 | 0.4314 | 0.3704 | 446,213 | 195.50 |
| lora | 11713 | 0.4823 | 0.4203 | 446,213 | 194.93 |
| reft | 11711 | 0.4296 | 0.3410 | 3,885 | 82.67 |
| reft | 11712 | 0.3697 | 0.3246 | 3,885 | 82.42 |
| reft | 11713 | 0.4541 | 0.3280 | 3,885 | 82.44 |

## Summary (mean ± std)
| Method | Dev Acc | Dev F1 | Trainable Params | Runtime (s) |
|---|---:|---:|---:|---:|
| full | 0.3896 ± 0.0057 | 0.3594 ± 0.0409 | 125,034,245 | 402.60 |
| last-linear | 0.4187 ± 0.0223 | 0.3579 ± 0.0502 | 3,845 | 73.91 |
| lora | 0.4553 ± 0.0256 | 0.4026 ± 0.0279 | 446,213 | 195.85 |
| reft | 0.4178 ± 0.0435 | 0.3312 ± 0.0086 | 3,885 | 82.51 |

## Analysis
- LoRA achieves the best average dev accuracy and F1, while keeping trainable parameters below 0.5M.
- Full finetuning is the slowest and most expensive, yet yields the weakest performance in this configuration.
- Last-linear and ReFT are extremely parameter-efficient and fast. Their accuracy is comparable, but ReFT shows higher variance across seeds.
- Overall, LoRA offers the best tradeoff between performance and efficiency for this task.

## Takeaways
- PEFT methods can outperform full finetuning while using far fewer parameters.
- LoRA is the most effective option in this experiment.
- ReFT may require additional tuning (e.g., layers or low-rank dimension) to match LoRA performance.
