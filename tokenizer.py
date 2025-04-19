from collections import Counter

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

from config import *


class Tokenizer:
    def __init__(self):
        self.tokenizer = get_tokenizer("basic_english")
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.vocab = None

    def build_vocab(self, texts):
        # counts token frequencies
        counter = Counter()
        for text in texts:
            tokens = self.tokenizer(text)
            counter.update(tokens)

        # apply min_freq
        filtered_tokens = {
            tok: freq for tok, freq in counter.items() if freq >= TOK_MIN_FREQ
        }
        specials = [self.pad_token, self.unk_token]

        # sort and truncate vocab
        sorted_by_freq = sorted(filtered_tokens.items(), key=lambda x: (-x[1], x[0]))
        sorted_by_freq = sorted_by_freq[: TOK_MAX_VOCAB_SIZE - len(specials)]

        # creat vocab
        vocab_tokens = specials + [tok for tok, _ in sorted_by_freq]
        vocab_counter = Counter({tok: 1 for tok in vocab_tokens})
        self.vocab = Vocab(vocab_counter, specials=specials)
        self.pad_index = self.vocab[self.pad_token]
        self.unk_index = self.vocab[self.unk_token]

    def encode(self, text):
        tokens = self.tokenizer(text)
        unpadded_tokens = [
            self.vocab[token] if token in self.vocab.stoi else self.unk_index
            for token in tokens
        ]
        # pad to TOK_MAX_LEN
        return unpadded_tokens[:TOK_MAX_LEN] + [self.vocab[self.pad_token]] * max(
            0, TOK_MAX_LEN - len(unpadded_tokens)
        )

    def decode(self, token_ids):
        return [self.vocab.itos[i] for i in token_ids]
