"""
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
"""

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
    SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW
from sacrebleu.metrics import CHRF

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
    """Your GPT-2 Model designed for sonnet generation."""

    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(
            model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # By default, fine-tune the full model. TODO: this is maybe not idea.
        for param in self.gpt.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        """
        This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
        not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
        not just the distribution over next tokens for the last token!
        """
        output = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = output["last_hidden_state"]
        logits = self.gpt.hidden_state_to_token(hidden_states)

        return logits

    def get_device(self):
        for param in self.gpt.parameters():
            return param.device

    @torch.no_grad()
    def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
        """
        Generates an original sonnet using top-p sampling and softmax temperature.

        TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
        In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
        there are many.
        """
        token_ids = encoding.to(self.get_device())  # 1, S
        attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(
            self.get_device()
        )

        newline_token_id = self.tokenizer.encode("\n")[0]
        min_new_lines = 11
        for _ in range(max_length):
            # Forward pass to get logits
            logits_sequence = self.forward(token_ids, attention_mask)  # 1, S, vocab
            logits_last_token = (
                logits_sequence[:, -1, :] / temperature
            )  # Apply temperature scaling # [1, vocab]

            current_newlines = (token_ids[0] == newline_token_id).sum().item()
            if current_newlines < (
                3 + min_new_lines
            ):  # prompt 3 line + 11 generated lineds
                logits_last_token[0, self.tokenizer.eos_token_id] = -float("inf")

            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(logits_last_token, dim=-1)  # [1, vocab]

            # Top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_mask = (
                cumulative_probs <= top_p
            )  # 例如 [[True, True, True, True, False, False]]
            top_p_mask[..., 1:] = top_p_mask[
                ..., :-1
            ].clone()  # Shift mask right for proper thresholding
            top_p_mask[..., 0] = True  # Always include the highest probability token
            filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
            filtered_probs /= filtered_probs.sum(
                dim=-1, keepdim=True
            )  # Normalize probabilities

            # Sample from filtered distribution
            sampled_index = torch.multinomial(filtered_probs, 1)
            sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

            # Stop if end-of-sequence token is reached
            if sampled_token.item() == self.tokenizer.eos_token_id:
                break

            # Append sampled token
            token_ids = torch.cat([token_ids, sampled_token], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((1, 1), dtype=torch.int64).to(self.get_device()),
                ],
                dim=1,
            )

        generated_output = self.tokenizer.decode(
            token_ids[0].cpu().tolist(), skip_special_tokens=True
        )
        return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def eval_loss(model, dataloader, device):
    model.eval()
    total_loss, num_batches = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            b_ids, b_mask = batch["token_ids"].to(device), batch["attention_mask"].to(
                device
            )
            logits = model(b_ids, b_mask)
            logits = rearrange(logits[:, :-1].contiguous(), "b t d -> (b t) d")
            labels = b_ids[:, 1:].clone()
            labels[~b_mask[:, 1:].bool()] = -100
            total_loss += F.cross_entropy(
                logits, labels.flatten(), ignore_index=-100
            ).item()
            num_batches += 1
    return total_loss / num_batches


@torch.no_grad()
def eval_chrf(
    model, prompt_dataset, reference_dataset, device, temperature=0.7, top_p=0.9
):
    model.eval()
    generated_texts, reference_texts = [], []
    for idx in range(len(prompt_dataset)):
        prompt_text = prompt_dataset[idx][1]
        encoding = model.tokenizer(
            prompt_text, return_tensors="pt", padding=False, truncation=True
        ).to(device)
        _, generated_text = model.generate(
            encoding["input_ids"], temperature=temperature, top_p=top_p
        )
        generated_texts.append(generated_text)
        reference_texts.append(reference_dataset[idx][1])
    return float(CHRF().corpus_score(generated_texts, [reference_texts]).score)


def train(args):
    """Train GPT-2 for sonnet generation on the Quora dataset."""
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # Create the data and its corresponding datasets and dataloader.
    sonnet_dataset = SonnetsDataset(args.sonnet_path)
    sonnet_dataloader = DataLoader(
        sonnet_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=sonnet_dataset.collate_fn,
    )

    # Validation set: complete sonnets for CE loss monitoring.
    val_dataset = SonnetsDataset(args.val_sonnet_path)
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=val_dataset.collate_fn,
    )

    # Val datasets for chrF early stopping: 3-line prompts + complete references.
    val_held_out_sonnet_dataset = SonnetsDataset(args.val_held_out_sonnet_path)
    val_true_sonnet_dataset = SonnetsDataset(args.val_sonnet_path)

    # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    args = add_arguments(args)
    model = SonnetGPT(args)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_val_chrf = -1.0
    epochs_without_improvement = 0
    best_model_path = f"best_{args.filepath}"

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(
            sonnet_dataloader, desc=f"train-{epoch}", disable=TQDM_DISABLE
        ):
            b_ids, b_mask = batch["token_ids"].to(device), batch["attention_mask"].to(
                device
            )

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            logits = rearrange(logits[:, :-1].contiguous(), "b t d -> (b t) d")
            labels = b_ids[:, 1:].clone()
            labels[~b_mask[:, 1:].bool()] = -100
            loss = F.cross_entropy(logits, labels.flatten(), ignore_index=-100)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches
        val_loss = eval_loss(model, val_dataloader, device)
        val_chrf = eval_chrf(
            model,
            val_held_out_sonnet_dataset,
            val_true_sonnet_dataset,
            device,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(
            f"Epoch {epoch}: train loss :: {train_loss:.3f}, val loss :: {val_loss:.3f}, val chrF :: {val_chrf:.2f}."
        )

        if val_chrf > best_val_chrf:
            best_val_chrf = val_chrf
            epochs_without_improvement = 0
            save_model(model, optimizer, args, best_model_path)
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement}/{args.patience} epoch(s). Best val chrF: {best_val_chrf:.2f}."
            )
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered after epoch {epoch}.")
                break

        print("Generating several output sonnets...")
        model.eval()
        for batch in held_out_sonnet_dataset:
            encoding = model.tokenizer(
                batch[1], return_tensors="pt", padding=True, truncation=True
            ).to(device)
            output = model.generate(
                encoding["input_ids"], temperature=args.temperature, top_p=args.top_p
            )
            print(f"{output[1]}\n\n")


@torch.no_grad()
def generate_submission_sonnets(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    saved = torch.load(f"best_{args.filepath}", weights_only=False)

    model = SonnetGPT(saved["args"])
    model.load_state_dict(saved["model"])
    model = model.to(device)
    model.eval()

    # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    generated_sonnets = []
    for batch in held_out_sonnet_dataset:  # 每一首诗
        sonnet_id = batch[0]
        encoding = model.tokenizer(
            batch[1], return_tensors="pt", padding=False, truncation=True
        ).to(device)
        output = model.generate(
            encoding["input_ids"], temperature=args.temperature, top_p=args.top_p
        )[0][0]
        decoded_output = model.tokenizer.decode(output)
        full_sonnet = f"{decoded_output}\n\n"
        generated_sonnets.append((sonnet_id, full_sonnet))

        print(f"{decoded_output}\n\n")  # 完整的一首诗，包括原来的三行

    with open(args.sonnet_out, "w+") as f:
        f.write(f"--Generated Sonnets-- \n\n")
        for sonnet in generated_sonnets:
            f.write(f"\n{sonnet[0]}\n")
            f.write(sonnet[1])


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
    parser.add_argument(
        "--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt"
    )
    parser.add_argument(
        "--sonnet_out", type=str, default="predictions/generated_sonnets.txt"
    )

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action="store_true")

    # Generation parameters.
    parser.add_argument(
        "--temperature", type=float, help="softmax temperature.", default=1.2
    )
    parser.add_argument(
        "--top_p",
        type=float,
        help="Cumulative probability distribution for nucleus sampling.",
        default=0.9,
    )

    parser.add_argument(
        "--batch_size", help="The training batch size.", type=int, default=8
    )
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument(
        "--patience",
        type=int,
        help="Early stopping patience (epochs without val chrF improvement).",
        default=3,
    )
    parser.add_argument(
        "--val_sonnet_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt"
    )
    parser.add_argument(
        "--val_held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        help="The model size as specified on hugging face.",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        default="gpt2",
    )

    args = parser.parse_args()
    return args


def add_arguments(args):
    """Add arguments that are deterministic on model size."""
    if args.model_size == "gpt2":
        args.d = 768
        args.l = 12
        args.num_heads = 12
    elif args.model_size == "gpt2-medium":
        args.d = 1024
        args.l = 24
        args.num_heads = 16
    elif args.model_size == "gpt2-large":
        args.d = 1280
        args.l = 36
        args.num_heads = 20
    else:
        raise Exception(f"{args.model_size} is not supported.")
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f"{args.epochs}-{args.lr}-sonnet.pt"  # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train(args)
    generate_submission_sonnets(args)
