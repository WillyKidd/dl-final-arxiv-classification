import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from config import TEST_PATH
from dataset import ArxivDataset
from model import ArxivClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = ArxivDataset(TEST_PATH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = ArxivClassifier(
    len(test_dataset.tokenizer.vocab),
    len(set(test_dataset.labels)),
    DIM_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    DIM_FFN,
).to(device)

model.load_state_dict(torch.load("model.pt"))
model.eval()

correct = total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        # Uncomment if doing joint encodings
        #input_ids = batch["input_ids"].to(device)
        #attention_mask = batch["attention_mask"].to(device)
        title_ids = batch["title_ids"].to(device)
        abstract_ids = batch["abstract_ids"].to(device)
        attention_mask_title = batch["attention_mask_title"].to(device)
        attention_mask_abstract = batch["attention_mask_abstract"].to(device)
        labels = batch["label"].to(device)

        # Same here as well if doing joint encodings
        #logits = model(input_ids, attention_mask)
        logits = model(title_ids, attention_mask_title, abstract_ids, attention_mask_abstract)

        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

accuracy = 100.0 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
