#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./encode.py <file|directory|glob> /path/to/output.npz
#  PYTHONPATH=src ./train --dataset /path/to/output.npz

import argparse
import numpy as np

import encoder
import encoder_sp
import encoder_hug
from load_dataset import load_dataset

parser = argparse.ArgumentParser(
    description="Pre-encode text files into tokenized training set.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "in_text",
    metavar="PATH",
    type=str,
    help="Input file, directory, or glob pattern (utf-8 text).",
)

parser.add_argument(
    "out_npz",
    metavar="OUT",
    type=str,
    help="Output file name."
)

parser.add_argument(
    "--model_name",
    metavar="MODEL",
    type=str,
    default="117M",
    help="Pretrained model name",
)

parser.add_argument(
    "--special_tokens",
    type=str,
    default="<|endoftext|>",
    help="Special predefined tokens. Separate with comma.",
)

parser.add_argument(
    "--combine",
    metavar="CHARS",
    type=int,
    default=50000,
    help="Concatenate files with <|endoftext|> separator into chunks of this minimum size",
)

parser.add_argument(
    "--encoding",
    type=str,
    default="utf-8",
    help="Set the encoding for reading and writing files.",
)


parser.add_argument(
    "--encoder",
    metavar="ENCODER",
    choices=["default", "sentencepiece", "huggingface"],
    default="default",
    type=str,
    help="Type of encoder. Choices: default, sentencepiece, huggingface. Default: default provided by OpenAI, src/encoder.py",
)



def main():
    args = parser.parse_args()
    if args.special_tokens:
        args.special_tokens = args.special_tokens.split(",")
    if args.encoder == "default":
        enc = encoder.get_encoder(args.model_name, "models", args.special_tokens)
    elif args.encoder == "sentencepiece":
        try:
            enc = encoder_sp.get_encoder(args.model_name, "models")
        except Exception as e:
            print("-"*40)
            print(e)
            exit("The SentencePiece model is not given by default by OpenAI. Try generate a new one using new_sp_model.py.")
    elif args.encoder == "huggingface":
        enc = encoder_hug.get_encoder(args.model_name, "models")
        if args.special_tokens:
            enc.tok.add_special_tokens(args.special_tokens)

    print("Reading files")
    chunks = load_dataset(enc, args.in_text, args.combine, encoding=args.encoding)
    print(f"Writing {args.out_npz}.npz")
    np.savez_compressed(args.out_npz, *chunks)


if __name__ == "__main__":
    main()
