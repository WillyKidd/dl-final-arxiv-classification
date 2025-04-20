import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from config import TEST_PATH
from dataset import ArxivDataset
from model import ArxivClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ArxivDataset(TRAIN_PATH)
test_dataset = ArxivDataset(TEST_PATH)
test_dataset.tokenizer = train_dataset.tokenizer
test_dataset.encoded = [
    test_dataset.tokenizer.encode(text) for text in test_dataset.texts
]
test_dataset.padded = test_dataset.tokenizer.pad_batch(test_dataset.encoded)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = ArxivClassifier(
    vocab_size=len(train_dataset.tokenizer.vocab),
    num_classes=len(train_dataset.labels),
).to(device)

model.load_state_dict(torch.load("model.pt"))
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

accuracy = 100.0 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
