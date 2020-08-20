"""Byte pair encoding utilities"""
import os
from tokenizers import ByteLevelBPETokenizer


class Encoder:
    def __init__(self, dirname, special_tokens=["<|endoftext|>"]):
        self.tok = ByteLevelBPETokenizer(
            vocab_file=os.path.join(dirname, "encoder.json"),
            merges_file=os.path.join(dirname, "vocab.bpe"),
        )
        if special_tokens:
            for s in special_tokens:
                assert (
                    s in self.tok.get_vocab()
                ), f"Special token '{s}' not found in encoder!"
            self.tok.add_special_tokens(special_tokens)

    def encode(self, text):
        return self.tok.encode(text).ids

    def decode(self, tokens):
        return self.tok.decode(tokens)


def get_encoder(model_name, models_dir="models", special_tokens=None):
    if special_tokens is not None:
        return Encoder(
            os.path.join(models_dir, model_name), special_tokens=special_tokens
        )
    else:
        return Encoder(os.path.join(models_dir, model_name,))
