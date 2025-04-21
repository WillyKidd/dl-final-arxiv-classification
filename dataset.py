import pandas as pd
import torch
from torch.utils.data import Dataset

from tokenizer import Tokenizer


class ArxivDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        self.texts = (self.df["title"] + " " + self.df["abstract"]).tolist()
        self.labels = self.df["label_id"].tolist()

        self.tokenizer = Tokenizer()

        encoded = [self.tokenizer.encode(text) for text in self.texts]
        self.padded = self.tokenizer.pad_batch(encoded)

    def __len__(self):
        return len(self.padded)

    def __getitem__(self, idx):
        input_ids = self.padded[idx]
        attention_mask = [
            1 if token != self.tokenizer.pad_index else 0 for token in input_ids
        ]
        label = self.labels[idx]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }
