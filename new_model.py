"""Generate new model vocab for fresh training using the ByteLevelPairEncoding by Huggingface"""
"""Do not forget to use PYTHONPATH=src python new_model.py ... [args]..."""
import os
import json
import glob
import shutil
import encoder
import argparse
import numpy as np
from load_dataset import load_dataset
from tokenizers import ByteLevelBPETokenizer

parser = argparse.ArgumentParser(
    description="""
    Create a fresh model using Hugginface's Tokenizer library.
    Train the BPE model on the dataset.
    Generate the npz file that will be use for training.
    Add the appropriate files to new model folder in "models"
    """,
    # formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--model_name", required=True, help="New model name. Required.")

parser.add_argument(
    "--source",
    required=True,
    help="""The source dataset, file or folder, glob allowed.
    Make sure you surround this parameter with quotes!""",
)

parser.add_argument(
    "--from_pretrained",
    default=None,
    help="""Pretrained model, e.g. '117M' or '355M'. If specified, the hparams
    will be imported. Otherwise, the user will be asked to choose new ones.""",
)

parser.add_argument(
    "--vocab_size",
    type=int,
    default=50257,
    help="The number of tokens in the vocab. Defaults to 50257.",
)

parser.add_argument(
    "--special_tokens",
    type=str,
    help="""Adding special markers in the vocabulary, e.g. markers. Separate by
    comma and enclose in quotes!""",
)

parser.add_argument(
    "--skip_encoding_npz",
    action="store_true",
    help="""Do not create a npz file of the dataset for training. Defaults to
    false.""",
)

parser.add_argument(
    "--combine",
    type=int,
    default=50000,
    help="Concatenate files with <|endoftext|> separator into chunks of this minimum size",
)

parser.add_argument("--encoding", type=str, default="utf-8", help="Default: utf-8.")


def underprint(x):
    print(x)
    print("-" * len(x))


def hparams_from_input():
    underprint("defining network params:")
    print("default params:")
    print(f"\tn_vocab: {args.vocab_size}")
    print("\tn_ctx: 1024")
    print("\tn_embd: 768")
    print("\tn_head: 12")
    print("\tn_layer: 12")
    answ = input("do you accept the default params? [y/n] ")
    if answ in ("y", "Y", "1"):
        hparams = {
            "n_vocab": 50257,
            "n_ctx": 1024,
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
        }
    else:
        hparams = {
            "n_vocab": questionnaire("vocab"),
            "n_ctx": questionnaire("attention window"),
            "n_embd": questionnaire("embedding"),
            "n_head": questionnaire("heads"),
            "n_layer": questionnaire("layers"),
        }

    fname = os.path.join("models", args.model_name, "hparams.json")
    with open(fname, "w") as o:
        print(f"written hparams to {fname})")
        json.dump(hparams, o)
    print()
    return hparams


def questionnaire(name):
    answ = input(f"please give the desired {name} size (int): ")
    while not int(answ):
        answ = input("please give a whole number (int): ")
    return abs(int(answ))


def train(args):
    model_dir = os.path.join("models", args.model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    fname = "hparams.json"
    if args.from_pretrained:
        source_params = os.path.join("models", args.from_pretrained, fname)
        shutil.copyfile(
            source_params, os.path.join(model_dir, fname),
        )
        with open(source_params) as i:
            hparams = json.load(i)
    else:
        hparams = hparams_from_input()
    underprint("chosen hparams:")
    for k, v in hparams.items():
        print(f"\t- {k}: {v}")
    print()

    underprint("training:")
    tok = ByteLevelBPETokenizer()
    fnames = [
        f
        for f in glob.glob(os.path.join(args.source, "**"), recursive=True)
        if os.path.isfile(f)
    ]
    tok.train(fnames, vocab_size=args.vocab_size, special_tokens=args.special_tokens)
    tok.save(directory=model_dir)
    os.rename(
        os.path.join(model_dir, "merges.txt"), os.path.join(model_dir, "vocab.bpe")
    )
    os.rename(
        os.path.join(model_dir, "vocab.json"), os.path.join(model_dir, "encoder.json")
    )

    if not args.skip_encoding_npz:
        underprint(f"generating npz encoding")
        enc = encoder.get_encoder(args.model_name, "models")
        chunks = load_dataset(enc, args.source, args.combine, args.encoding)
        source_name = os.path.splitext(os.path.basename(args.source))[0]
        out_name = f"{source_name}-{args.model_name}"
        print(f"writing {out_name}.npz")
        np.savez_compressed(f"{out_name}", *chunks)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.special_tokens:
        args.special_tokens = args.special_tokens.split(",")
    else:
        args.special_tokens = []
    train(args)
