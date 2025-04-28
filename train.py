# import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset import ArxivDataset
from model import ArxivClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ArxivDataset(TRAIN_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = ArxivDataset(TEST_PATH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = ArxivClassifier(
    len(dataset.tokenizer.vocab),
    len(set(dataset.labels)),
    DIM_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    DIM_FFN,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# (for csv)
results = [["Epochs", "Training Accuracy", "Testing Accuracy"]]

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

    # Below for testing (per epoch) and updating csv file

    training_accuracy = 100.0 * correct / total

    model.eval()

    correct = total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    testing_accuracy = 100.0 * correct / total

    results.append([epoch + 1, training_accuracy, testing_accuracy])

torch.save(model.state_dict(), "model.pt")

# Adding the results to the csv
with open("results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(results)