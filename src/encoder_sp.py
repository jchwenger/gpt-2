"""Byte pair encoding utilities"""
import os
import sentencepiece as spm


class Encoder:
    def __init__(self, filename):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(filename)

    def encode(self, text):
        return self.sp.EncodeAsIds(text)

    def decode(self, tokens):
        return self.sp.DecodeIds(tokens.tolist())


def get_encoder(models_dir, model_name):
    return Encoder(os.path.join(models_dir, model_name, "sp.model"))
