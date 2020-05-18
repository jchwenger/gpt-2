"""Create new model vocab for fresh training using Sentencepiece"""
import os
import glob
import argparse
import numpy as np
from shutil import copyfile
import sentencepiece as spm
import encoder_sp as encoder_sp
from load_dataset import load_dataset


parser = argparse.ArgumentParser(
    description="""
    Create a fresh model using SentencePiece.
    Train a SentencePiece model on the dataset.
    Generate the npz file that will be use for training.
    Add the appropriate files to new model folder in "models"
    (Copying the pretrained model checkpoint as a starting ground.)
    """,
    # formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--source",
    required=True,
    help="""The source dataset, file or folder, glob allowed.
    Make sure you surround this parameter with quotes!"""
)

parser.add_argument(
    "--pretrained_model",
    default="117M",
    help="Pretrained model. Default 117M."
)

parser.add_argument(
    "--model_name",
    required=True,
    help="New model name. Required."
)

parser.add_argument(
    "--sp_model_name",
    type=str,
    default="sp",
    help="File name prefix for .vocab and .model. Default: sp."
)

parser.add_argument(
    "--encoding",
    type=str,
    default="utf-8",
    help="Default: utf-8."
)

parser.add_argument(
    "--vocab_size",
    type=int,
    default=50000,
    help="Default: 50001."
)

parser.add_argument(
    "--character_coverage",
    type=float,
    default=1.0,
    help="Char coverage, default: 1.0 (100%%) for non-ideogram languages."
)

parser.add_argument(
    "--sp_model_type",
    type=str,
    default="bpe",
    help="Byte Pair Encoding as default."
)

parser.add_argument(
    "--input_sentence_size",
    type=int,
    default=1000000,
    help="""Number of sentences to be sampled during sp training. Defaults to 1000000.
    """
)

parser.add_argument(
    "--shuffle_input_sentence",
    type=bool,
    default=True,
    help="""Shuffle input sentences. Defaults to true."""
)

parser.add_argument(
    "--control_symbols",
    type=str,
    help="""List of specific markers present in the dataset.
    They will always be encoded as one token.
    """
)

parser.add_argument(
    "--skip_encoding_npz",
    action='store_true',
    help="Create a npz file of the dataset for training."
)

parser.add_argument(
    "--combine",
    type=int,
    default=50000,
    help="Concatenate files with <|endoftext|> separator into chunks of this minimum size"
)

def train_sp(args):

    if not os.path.isdir(os.path.join("models", args.model_name)):
        os.mkdir(os.path.join("models", args.model_name))

    paths = []
    # ported from load_dataset
    if os.path.isfile(args.source):
        # Simple file
        paths.append(args.source)
    elif os.path.isdir(args.source):
        # Directory
        for (dirpath, _, fnames) in os.walk(args.source):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(args.source)

    paths = ",".join(paths)
    cmd = f"--input={paths}\
              --model_prefix={args.sp_model_name}\
              --vocab_size={args.vocab_size}\
              --character_coverage={args.character_coverage}\
              --model_type={args.sp_model_type}\
              --input_sentence_size={args.input_sentence_size}\
              --shuffle_input_sentence={args.shuffle_input_sentence}"
    if args.control_symbols:
        cmd += f" --control_symbols={args.control_symbols}"
    spm.SentencePieceTrainer.Train(cmd)

    print_separator()
    print(f"moving files to models/{args.model_name}")
    for suffix in (".model", ".vocab"):
        os.rename(
            f"{args.sp_model_name}{suffix}",
            os.path.join("models", args.model_name, f"{args.sp_model_name}{suffix}"),
        )

    print(f"generating hparams.json in models/{args.model_name}")
    hparams = '{ "n_vocab": 50257, "n_ctx": 1024, "n_embd": 768, "n_head": 12, "n_layer": 12 }'

    with open(os.path.join("models", args.model_name, "hparams.json"), "w") as o:
        o.write(hparams)

    print(f"copying checkpoint from pretrained models/{args.pretrained_model}")
    src_dir = os.path.join("models", args.pretrained_model)
    dest_dir = os.path.join("models", args.model_name)
    ckpt_fnames = [f for f in os.listdir(src_dir) if "model" in f or "checkpoint" in f]
    for f in ckpt_fnames:
        copyfile(os.path.join(src_dir, f), os.path.join(dest_dir, f))

    if not args.skip_encoding_npz:
        print_separator()
        print(f"generating encoding")
        enc = encoder_sp.get_encoder(args.model_name, "models")
        chunks = load_dataset(enc, args.source, args.combine, args.encoding)
        print(f"writing {args.model_name}-{args.pretrained_model}-sp.npz")
        np.savez_compressed(f"{args.model_name}-{args.pretrained_model}-sp", *chunks)


def print_separator():
    print("-" * 40)


if __name__ == "__main__":
    args = parser.parse_args()
    train_sp(args)
