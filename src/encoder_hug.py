"""Byte pair encoding utilities"""
import os
from tokenizers import ByteLevelBPETokenizer


class Encoder:
    def __init__(self, dirname):
        self.tok = ByteLevelBPETokenizer(
        vocab_file=os.path.join(dirname, "encoder.json"),
        merges_file=os.path.join(dirname, "vocab.bpe"),
        )

    def encode(self, text):
        return self.tok.encode(text).ids

    def decode(self, tokens):
        return self.tok.decode(tokens)


def get_encoder(model_name, models_dir="models"):
    return Encoder(os.path.join(models_dir, model_name))
