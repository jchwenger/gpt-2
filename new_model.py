"""Generate new model vocab for fresh training using the ByteLevelPairEncoding by Huggingface"""
"""Do not forget to use PYTHONPATH=src python new_model.py ... [args]..."""
import os
import json
import glob
import shutil
import argparse
import encoder_hug
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
    "--precision",
    type=str,
    choices=["float32", "float16", "bfloat16"],
    default="float32",
    help="The precision to use for the network. Choices: float32, float16, bfloat16 (TPUs). Defaults to float32.",
)

parser.add_argument(
    "--special_tokens",
    default="<|endoftext|>",
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
    help="""Concatenate files with <|endoftext|> separator into chunks of this
    minimum size. Defaults to: 50000.""",
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
    print("\tprecision: float32")
    answ = input("do you accept the default params? [y/n] ")
    hparams = {
        "n_vocab": 50257,
        "n_ctx": 1024,
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 12,
        "le_dtype": "float32",
    }
    if answ not in ("y", "Y", "1"):
        hparams = {
            "n_vocab": questionnaire("vocab", hparams["n_vocab"]),
            "n_ctx": questionnaire("attention window", hparams["n_ctx"]),
            "n_embd": questionnaire("embedding", hparams["n_embd"]),
            "n_head": questionnaire("heads", hparams["n_head"]),
            "n_layer": questionnaire("layers", hparams["n_layer"]),
            "le_dtype": questionnaire("precision", hparams["le_dtype"]),
        }

    fname = os.path.join("models", args.model_name, "hparams.json")
    with open(fname, "w") as o:
        print(f"written hparams to {fname})")
        json.dump(hparams, o)
    print()
    return hparams


def questionnaire(name, hparam):
    if name == "precision":
        answ = input(
            f"desired precision, float32, float16, or bfloat16 (press enter for default): "
        )
        if answ == "":
            return hparam
        while answ not in ["float32", "float16", "bfloat16"]:
            answ = input(
                "please type the correct precision (float32, float16, or bfloat16)"
            )
        return answ
    else:
        answ = input(f"the desired {name} size (int) (press enter for default): ")
        if answ == "":
            return hparam
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
        enc = encoder_hug.get_encoder(args.model_name, "models")
        enc.tok.add_special_tokens(args.special_tokens)
        chunks = load_dataset(enc, args.source, args.combine, args.encoding)
        # a txt file, we take only its name
        if args.source.endswith(".txt"):
            source_name = os.path.splitext(os.path.basename(args.source))[0]
        elif args.source.endswith("/"):  # a dir ending in /
            source_name = os.path.basename(args.source[:-1])
        else:  # just a dir
            source_name = os.path.basename(args.source)
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
