import pandas as pd
import torch
from torch.utils.data import Dataset

from tokenizer import Tokenizer


class ArxivDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        # Encoding title and abstract together
        self.texts = (self.df["title"] + " " + self.df["abstract"]).tolist()

        # Encoding title and abstract separately
        self.titles = self.df["title"].tolist()
        self.abstracts = self.df["abstract"].tolist()

        self.labels = self.df["label_id"].tolist()

        self.tokenizer = Tokenizer()

        # Uncomment if you want to encode together
        #encoded = [self.tokenizer.encode(text) for text in self.texts]
        #self.padded = self.tokenizer.pad_batch(encoded)

        # Encoding titles and abstract separately
        encoded_titles = [self.tokenizer.encode(title) for title in self.titles]
        encoded_abstracts = [self.tokenizer.encode(abstract) for abstract in self.abstracts]

        self.padded_titles = self.tokenizer.pad_batch(encoded_titles)
        self.padded_abstracts = self.tokenizer.pad_batch(encoded_abstracts)

    def __len__(self):
        return len(self.padded)

    def __getitem__(self, idx):
        # Uncomment if you want to encode title and abstract together
        #input_ids = self.padded[idx]
        #attention_mask = [
        #    1 if token != self.tokenizer.pad_index else 0 for token in input_ids
        #]
        #label = self.labels[idx]
        #
        #return {
        #    "input_ids": torch.tensor(input_ids, dtype=torch.long),
        #    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        #    "label": torch.tensor(label, dtype=torch.long),
        #}

        # Encoding title and abstract separately
        title_ids = self.padded_titles[idx]
        abstract_ids = self.padded_abstracts[idx]

        attention_mask_title = [1 if token != self.tokenizer.pad_index else 0 for token in title_ids]
        attention_mask_abstract = [1 if token != self.tokenizer.pad_index else 0 for token in abstract_ids]
        label = self.labels[idx]

        return {
            "title_ids": torch.tensor(title_ids, dtype=torch.long),
            "abstract_ids": torch.tensor(abstract_ids, dtype=torch.long),
            "attention_mask_title": torch.tensor(attention_mask_title, dtype=torch.long),
            "attention_mask_abstract": torch.tensor(attention_mask_abstract, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }
