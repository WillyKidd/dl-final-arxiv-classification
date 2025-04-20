import json
import os
import zipfile

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

DOWNLOAD_URL = (
    "https://github.com/ashfarhangi/Protoformer/raw/refs/heads/main/data/ArXiv-10.zip"
)
CSV_PATH = "data/arxiv100.csv"
DATA_DIR = "data"


def download():
    zip_path = "ArXiv-10.zip"

    if os.path.exists(CSV_PATH):
        print("Files exist, skipping download and extraction.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading...")
    response = requests.get(DOWNLOAD_URL)
    with open(zip_path, "wb") as f:
        f.write(response.content)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    os.remove(zip_path)


def preprocess_and_split(
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1"

    os.makedirs(DATA_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    labels = sorted(df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    df["label_id"] = df["label"].map(label2id)

    train_df, temp_df = train_test_split(
        df, train_size=train_ratio, stratify=df["label_id"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_df["label_id"],
        random_state=42,
    )

    cols_to_save = ["title", "abstract", "label", "label_id"]
    train_df[cols_to_save].to_csv(f"{DATA_DIR}/train.csv", index=False)
    val_df[cols_to_save].to_csv(f"{DATA_DIR}/val.csv", index=False)
    test_df[cols_to_save].to_csv(f"{DATA_DIR}/test.csv", index=False)

    with open(f"{DATA_DIR}/label2id.json", "w") as f:
        json.dump(label2id, f, indent=2)
    with open(f"{DATA_DIR}/id2label.json", "w") as f:
        json.dump(id2label, f, indent=2)

    print(f"Preprocess complete, files saved to {DATA_DIR}")


if __name__ == "__main__":
    download()
    preprocess_and_split()
