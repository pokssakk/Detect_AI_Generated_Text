import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from torch.optim import AdamW
import pandas as pd
import yaml
from tqdm import tqdm

from model import RobertaClassifier
from dataset import ParagraphDataset
from utils import load_config, get_device, load_tokenizer

config = load_config()
device = get_device()
print("Using device:", device)

tokenizer = load_tokenizer(config['model']['name'])
train_dataset = ParagraphDataset(
    csv_path = config['data']['train_path'], 
    tokenizer = tokenizer,
    max_len = config['model']['max_length'],
    stride=config['model']['stride'],
    cache_path=config['data']['train_cache_path']
)

train_loader = DataLoader(
    train_dataset, 
    batch_size = config['train']['batch_size'],
    shuffle = True
)

print("Total train batches:", len(train_loader))

model = RobertaClassifier(
    model_name=config['model']['name'],
    dropout = config['model']['dropout']
).to(device)

optimizer = AdamW(
    model.parameters(), 
    lr = float(config['train']['lr']),
    weight_decay = config['train']['weight_decay']
)

num_training_steps = config['train']['epochs'] * len(train_loader)

lr_scheduler = get_scheduler(
    name = "linear", 
    optimizer = optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

losses = []

for epoch in range(config['train']['epochs']):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for step, batch in enumerate(loop):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"\nEpoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

save_path = config['model']['save_path']
torch.save(model.state_dict(), save_path)
print(f"모델 '{save_path}'에 저장.")


import matplotlib.pyplot as plt

plt.plot(range(1, len(losses)+1), losses, marker = 'o')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()