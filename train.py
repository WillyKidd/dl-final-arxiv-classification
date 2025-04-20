import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset import ArxivDataset
from model import ArxivClassifier
from tokenizer import Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(TRAIN_PATH)
texts = (df["title"] + " " + df["abstract"]).tolist()
labels = sorted(df["label"].unique())

tokenizer = Tokenizer()
tokenizer.build_vocab(texts)

NUM_CLASSES = len(labels)

dataset = ArxivDataset(TRAIN_PATH, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ArxivClassifier(
    vocab_size=len(tokenizer.vocab),
    num_classes=NUM_CLASSES,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            loss=running_loss / (total / BATCH_SIZE), acc=100.0 * correct / total
        )

# 8. Save model and label map
torch.save(model.state_dict(), "model.pt")
print("✅ Training complete and model saved.")
