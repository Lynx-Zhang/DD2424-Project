#!/usr/bin/env python3

'''
Trains and evaluates GPT2SentimentClassifier with PEFT (LoRA/ReFT)
'''

import random, numpy as np, argparse
from types import SimpleNamespace
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.metrics import f1_score, accuracy_score

from models.gpt2 import GPT2Model
from optimizer import AdamW
from tqdm import tqdm
from peft_utils import get_lora_model, get_reft_model, count_trainable_parameters

TQDM_DISABLE = False

def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

class GPT2SentimentClassifier(torch.nn.Module):
  def __init__(self, config):
    super(GPT2SentimentClassifier, self).__init__()
    self.num_labels = config.num_labels
    self.gpt = GPT2Model.from_pretrained()

    # Freeze all parameters first
    for param in self.gpt.parameters():
      param.requires_grad = False

    # Apply PEFT
    if config.peft_type == "lora":
      target_modules = getattr(config, "lora_targets", ["query", "key", "value"])
      self.gpt = get_lora_model(
        self.gpt,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
      )
    elif config.peft_type == "reft":
        # Example: intervene on last 2 layers
        layers = list(range(max(0, self.gpt.config.num_hidden_layers - config.reft_layers), self.gpt.config.num_hidden_layers))
        self.gpt = get_reft_model(self.gpt, r=config.reft_r, layers=layers, positions=config.reft_positions)
    elif config.peft_type == "full":
        for param in self.gpt.parameters():
            param.requires_grad = True

    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
    self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels) 
    
    print(f"Total trainable parameters: {count_trainable_parameters(self)}")

  def forward(self, input_ids, attention_mask):
      output = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
      last_token = output['last_token']
      logits = self.classifier(self.dropout(last_token))
      return logits

# Reusing SentimentDataset from classifier.py (simplified import or copy)
class SentimentDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def pad_data(self, data):
    sents = [x[0] for x in data]
    labels = [x[1] for x in data]
    sent_ids = [x[2] for x in data]
    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])
    labels = torch.LongTensor(labels)
    return token_ids, attention_mask, labels, sents, sent_ids

  def collate_fn(self, all_data):
    token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)
    return {'token_ids': token_ids, 'attention_mask': attention_mask, 'labels': labels, 'sents': sents, 'sent_ids': sent_ids}

def load_data(filename, flag='train'):
  num_labels = {}
  data = []
  with open(filename, 'r') as fp:
    for record in csv.DictReader(fp, delimiter='\t'):
      sent = record['sentence'].lower().strip()
      sent_id = record['id'].lower().strip()
      if flag == 'test':
        data.append((sent, sent_id))
      else:
        label = int(record['sentiment'].strip())
        if label not in num_labels:
          num_labels[label] = len(num_labels)
        data.append((sent, label, sent_id))
  if flag == 'train': return data, len(num_labels)
  return data

def model_eval(dataloader, model, device):
  model.eval()
  y_true, y_pred = [], []
  for batch in tqdm(dataloader, desc='eval', disable=TQDM_DISABLE):
    b_ids, b_mask, b_labels = batch['token_ids'].to(device), batch['attention_mask'].to(device), batch['labels']
    logits = model(b_ids, b_mask)
    preds = np.argmax(logits.detach().cpu().numpy(), axis=1).flatten()
    y_true.extend(b_labels.flatten())
    y_pred.extend(preds)
  return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')

def train(args):
  device = torch.device('cuda') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
  train_data, num_labels = load_data(args.train, 'train')
  dev_data = load_data(args.dev, 'valid')
  train_dataloader = DataLoader(SentimentDataset(train_data, args), shuffle=True, batch_size=args.batch_size, collate_fn=SentimentDataset(train_data, args).collate_fn)
  dev_dataloader = DataLoader(SentimentDataset(dev_data, args), shuffle=False, batch_size=args.batch_size, collate_fn=SentimentDataset(dev_data, args).collate_fn)

  config = SimpleNamespace(
      peft_type=args.peft_type, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
      reft_r=args.reft_r, reft_layers=args.reft_layers, reft_positions=args.reft_positions,
      hidden_dropout_prob=args.hidden_dropout_prob, num_labels=num_labels, hidden_size=768
  )
  model = GPT2SentimentClassifier(config).to(device)
  optimizer = AdamW(model.parameters(), lr=args.lr)
  best_dev_acc = 0

  for epoch in range(args.epochs):
    model.train()
    for batch in tqdm(train_dataloader, desc=f'train-{epoch}'):
      b_ids, b_mask, b_labels = batch['token_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
      loss.backward()
      optimizer.step()
    
    train_acc, _ = model_eval(train_dataloader, model, device)
    dev_acc, _ = model_eval(dev_dataloader, model, device)
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        # torch.save(model.state_dict(), args.filepath)
    print(f"Epoch {epoch}: train acc :: {train_acc:.3f}, dev acc :: {dev_acc:.3f}")
  return best_dev_acc

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--peft_type", choices=['lora', 'reft', 'full', 'none'], default='lora')
  parser.add_argument("--lora_r", type=int, default=8)
  parser.add_argument("--lora_alpha", type=int, default=16)
  parser.add_argument("--reft_r", type=int, default=4)
  parser.add_argument("--reft_layers", type=int, default=2)
  parser.add_argument("--reft_positions", choices=['all', 'last', 'first'], default='last')
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
  parser.add_argument("--train", default='data/ids-sst-train.csv')
  parser.add_argument("--dev", default='data/ids-sst-dev.csv')
  args = parser.parse_args()
  
  seed_everything(args.seed)
  train(args)
