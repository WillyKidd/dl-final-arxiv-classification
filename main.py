from tokenizer import Tokenizer
import pandas as pd
import torch


df = pd.read_csv("data/arxiv100.csv")
texts = (df['title'] + ' ' + df['abstract']).tolist()

tokenizer = Tokenizer()
tokenizer.build_vocab(texts)

encoded = tokenizer.encode("A new method for computing QCD observables")
print("Token IDs:", encoded)
print("Decoded:", tokenizer.decode(encoded))

batch = [tokenizer.encode(t) for t in texts[:4]]
