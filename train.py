#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./train --dataset <file|directory|glob>

import os
import json
import time
import tqdm
import regex
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2

import model, sample, encoder
import encoder_sp as encoder_sp
import memory_saving_gradients
from accumulate import AccumulatingOptimizer
from load_dataset import load_dataset, Sampler
from adafactor_optimizer import AdafactorOptimizer
from adafactor_optimizer import adafactor_decay_rate_adam
from adafactor_optimizer import adafactor_decay_rate_pow

CHECKPOINT_DIR = "checkpoint"
SAMPLE_DIR = "samples"


parser = argparse.ArgumentParser(
    description="Fine-tune GPT-2 on your custom dataset.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--dataset",
    metavar="PATH",
    type=str,
    required=True,
    help="Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).",
)

parser.add_argument(
    "--model_name",
    metavar="MODEL",
    type=str,
    default="117M",
    help="Pretrained model name",
)

parser.add_argument(
    "--combine",
    metavar="CHARS",
    type=int,
    default=50000,
    help="Concatenate input files with <|endoftext|> separator into chunks of this minimum size",
)

parser.add_argument(
    "--encoding",
    type=str,
    default="utf-8",
    help="Set the encoding for reading and writing files.",
)

parser.add_argument(
    "--batch_size", metavar="SIZE", type=int, default=1, help="Batch size"
)

parser.add_argument(
    "--learning_rate",
    metavar="LR",
    type=float,
    default=0.1,
    help="Learning rate for Adam",
)

parser.add_argument(
    "--accumulate_gradients",
    metavar="N",
    type=int,
    default=1,
    help="Accumulate gradients across N minibatches.",
)

parser.add_argument(
    "--memory_saving_gradients",
    default=False,
    action="store_true",
    help="Use gradient checkpointing to reduce vram usage.",
)

parser.add_argument(
    "--only_train_transformer_layers",
    default=False,
    action="store_true",
    help="Restrict training to the transformer blocks.",
)

parser.add_argument(
    "--optimizer", type=str, default="adam", help="Optimizer. <adam|adafactor|sgd>."
)

parser.add_argument(
    "--decay_type",
    type=str,
    default="pow",
    help="""Decay type for optimizer, used with AdaFactor.
    Defaults to pow. <adam|pow>.""",
)

parser.add_argument(
    "--decay_lr_every",
    type=int,
    default=0,
    help="""Number of steps after which the exponential decay of the learning
    rate is applied. Defaults to 0 (no decay).""",
)

parser.add_argument(
    "--weight_decay",
    action="store_true",
    help="""Enable weight decay (for Adafactor, Adam)""",
)

parser.add_argument(
    "--noise",
    type=float,
    default=0.0,
    help="Add noise to input training data to regularize against typos.",
)

parser.add_argument("--top_k", type=int, default=0, help="K for top-k sampling.")

parser.add_argument(
    "--top_p",
    type=float,
    default=0.0,
    help="P for top-p sampling. Overrides top_k if set > 0.",
)

parser.add_argument(
    "--restore_from",
    type=str,
    default="latest",
    help='Either "latest", "fresh", or a path to a checkpoint file',
)
parser.add_argument(
    "--run_name",
    type=str,
    default="run1",
    help="Run id. Name of subdirectory in checkpoint/ and samples/",
)

parser.add_argument(
    "--sample_every",
    metavar="N",
    type=int,
    default=1000,
    help="Generate samples every N steps",
)

parser.add_argument(
    "--sample_length",
    metavar="TOKENS",
    type=int,
    default=1023,
    help="Sample this many tokens",
)

parser.add_argument(
    "--sample_num", metavar="N", type=int, default=1, help="Generate this many samples"
)

parser.add_argument(
    "--save_every",
    metavar="N",
    type=int,
    default=10000,
    help="Write a checkpoint every N steps",
)

parser.add_argument(
    "--val_dataset",
    metavar="PATH",
    type=str,
    default=None,
    help="Dataset for validation loss, defaults to --dataset.",
)

parser.add_argument(
    "--val_batch_size",
    metavar="SIZE",
    type=int,
    default=2,
    help="Batch size for validation.",
)

parser.add_argument(
    "--val_batch_count",
    metavar="N",
    type=int,
    default=40,
    help="Number of batches for validation.",
)

parser.add_argument(
    "--val_every",
    metavar="STEPS",
    type=int,
    default=0,
    help="Calculate validation loss every STEPS steps.",
)

parser.add_argument(
    "--encoder",
    # metavar="ENCODER",
    choices=["default", "sentencepiece"],
    default="default",
    type=str,
    help="Type of encoder. Choices: default, sentencepiece. Default: default",
)

parser.add_argument(
    "--reverse", action="store_true", help="Train on reversed token sequences",
)

parser.add_argument(
    "--print_train_sample",
    action="store_true",
    help="""Print the start of the training sample at each step.""",
)


def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass


def randomize(context, hparams, p):
    if p > 0:
        mask = tf.random.uniform(shape=tf.shape(context)) < p
        noise = tf.random.uniform(
            shape=tf.shape(context), minval=0, maxval=hparams.n_vocab, dtype=tf.int32
        )
        return tf.where(mask, noise, context)
    else:
        return context


def main():

    args = parser.parse_args()

    if args.encoder == "default":
        enc = encoder.get_encoder(args.model_name, "models")
    elif args.encoder == "sentencepiece":
        enc = encoder_sp.get_encoder(args.model_name, "models")

    hparams = model.default_hparams()
    with open(os.path.join("models", args.model_name, "hparams.json")) as f:
        hparams.override_from_dict(json.load(f))

    if args.sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx
        )

    config = tf.compat.v1.ConfigProto(allow_soft_placement=True) # , log_device_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    config.graph_options.rewrite_options.layout_optimizer = (
        rewriter_config_pb2.RewriterConfig.OFF
    )

    context = tf.compat.v1.placeholder(tf.int32, [args.batch_size, None])
    context_in = randomize(context, hparams, args.noise) if args.noise else context
    output = model.model(hparams=hparams, X=context_in)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=context[:, 1:], logits=output["logits"][:, :-1]
        )
    )

    if args.val_every > 0:
        val_context = tf.compat.v1.placeholder(
            tf.int32, [args.val_batch_size, None]
        )
        val_output = model.model(hparams=hparams, X=val_context)
        val_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=val_context[:, 1:], logits=val_output["logits"][:, :-1]
            )
        )
        val_loss_summary = tf.summary.scalar("val_loss", val_loss)

    tf_sample = sample.sample_sequence(
        hparams=hparams,
        length=args.sample_length,
        context=context,
        batch_size=args.batch_size,
        temperature=1.0,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    all_vars = [v for v in tf.compat.v1.trainable_variables() if "model" in v.name]
    train_vars = (
        [v for v in all_vars if "/h" in v.name]
        if args.only_train_transformer_layers
        else all_vars
    )

    counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, "counter")
    if os.path.exists(counter_path):
        # Load the step number if we're resuming a run
        # Add 1 so we don't immediately try to save again
        with open(counter_path, "r") as i:
            global_step = tf.Variable(
                int(i.read()), trainable=False, name="global_step"
            )
    else:
        global_step = tf.compat.v1.train.get_or_create_global_step()

    if args.decay_lr_every > 0:
        learning_rate = tf.compat.v1.train.exponential_decay(
            args.learning_rate, global_step, args.decay_lr_every, 0.96, staircase=True
        )
    else:
        learning_rate = tf.constant(args.learning_rate)

    if args.optimizer == "adam":
        if args.weight_decay:
            opt = tf.contrib.opt.AdamWOptimizer(
                learning_rate=learning_rate,
                weight_decay=0.01 * learning_rate,
                beta1=0.9,
                beta2=0.98,
                epsilon=1e-9,
            )
        else:
            opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    elif args.optimizer == "sgd":
        opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif args.optimizer == "adafactor":
        if args.decay_type == "adam":
            decay_rate = adafactor_decay_rate_adam(0.98)
        elif args.decay_type == "pow":
            decay_rate = adafactor_decay_rate_pow(0.8)
        else:
            raise ValueError("unknown optimizer_adafactor_decay_type")
        if args.weight_decay:
            AdafactorWOptimizer = tf.contrib.opt.extend_with_decoupled_weight_decay(
                AdafactorOptimizer
            )
            opt = AdafactorWOptimizer(
                weight_decay=0.01 * learning_rate,
                learning_rate=learning_rate,
                decay_rate=decay_rate,
                beta1=0.0,
                name="AdafactorW",
            )
        else:
            opt = AdafactorOptimizer(
                learning_rate=learning_rate,
                decay_rate=decay_rate,
                beta1=0.0,
                name="Adafactor",
            )
    else:
        exit("Bad optimizer:", args.optimizer)

    if args.accumulate_gradients > 1:
        if args.memory_saving_gradients:
            exit(
                "Memory saving gradients are not implemented for gradient accumulation yet."
            )
        opt = AccumulatingOptimizer(opt=opt, var_list=train_vars)
        opt_reset = opt.reset()
        opt_compute = opt.compute_gradients(loss)
        opt_apply = opt.apply_gradients(global_step=global_step)
        summary_loss = tf.compat.v1.summary.scalar("loss", opt_apply)
    else:
        if args.memory_saving_gradients:
            opt_grads = memory_saving_gradients.gradients(loss, train_vars)
        else:
            opt_grads = tf.gradients(loss, train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads, global_step=global_step)
        summary_loss = tf.compat.v1.summary.scalar("loss", loss)

    summary_lr = tf.compat.v1.summary.scalar("learning_rate", learning_rate)
    summaries = tf.compat.v1.summary.merge([summary_lr, summary_loss])

    summary_log = tf.compat.v1.summary.FileWriter(
        os.path.join(CHECKPOINT_DIR, args.run_name)
    )

    saver = tf.compat.v1.train.Saver(var_list=all_vars, max_to_keep=1)

    if args.restore_from == "latest":
        ckpt = tf.train.latest_checkpoint(
            os.path.join(CHECKPOINT_DIR, args.run_name)
        )
        if ckpt is None:
            # Get fresh GPT weights if new run.
            ckpt = tf.train.latest_checkpoint(
                os.path.join("models", args.model_name)
            )
    elif args.restore_from == "fresh":
        ckpt = tf.train.latest_checkpoint(os.path.join("models", args.model_name))
    else:
        ckpt = tf.train.latest_checkpoint(args.restore_from)

    print("Loading dataset...")
    chunks = load_dataset(enc, args.dataset, args.combine, encoding=args.encoding)
    if args.reverse:
        print("Reversing dataset...")
        chunks = [c[::-1] for c in chunks]
    data_sampler = Sampler(chunks)
    if args.val_every > 0:
        if args.val_dataset:
            val_chunks = load_dataset(
                enc, args.val_dataset, args.combine, encoding=args.encoding
            )
            if args.reverse:
                print("Reversing val dataset...")
                val_chunks = [c[::-1] for c in val_chunks]
        else:
            val_chunks = chunks
    print("dataset has", data_sampler.total_size, "tokens")

    if args.val_every > 0:
        # Sample from validation set once with fixed seed to make
        # it deterministic during training as well as across runs.
        val_data_sampler = Sampler(val_chunks, seed=1)
        val_batches = [
            [
                val_data_sampler.sample(hparams.n_ctx)
                for _ in range(args.val_batch_size)
            ]
            for _ in range(args.val_batch_count)
        ]

    with tf.compat.v1.Session(config=config) as sess:

        sess.run(global_step.initializer)
        sess.run(tf.compat.v1.global_variables_initializer())

        if ckpt is not None:
            print("-" * 40)
            print("Loading checkpoint", ckpt)
            saver.restore(sess, ckpt)

        print("-" * 40)
        print(
            f"Training... (global step: {sess.run(tf.compat.v1.train.get_global_step())})"
        )

        def save():
            maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
            gs = sess.run(tf.compat.v1.train.get_global_step())
            print(
                "Saving",
                os.path.join(CHECKPOINT_DIR, args.run_name, "model-{}").format(gs),
            )
            saver.save(
                sess,
                os.path.join(CHECKPOINT_DIR, args.run_name, "model"),
                global_step=gs,
            )
            with open(counter_path, "w") as o:
                o.write(str(gs) + "\n")

        def generate_samples():
            print("Generating samples...")
            context_tokens = data_sampler.sample(1)
            print("Sampled token used as seed:", enc.decode(context_tokens))
            all_text = []
            index = 0
            while index < args.sample_num:
                out = sess.run(
                    tf_sample, feed_dict={context: args.batch_size * [context_tokens]}
                )
                for i in range(min(args.sample_num - index, args.batch_size)):
                    if args.reverse:
                        textr = enc.decode(out[i])
                        text = enc.decode(out[i][::-1])
                    else:
                        text = enc.decode(out[i])
                    text = "======== SAMPLE {} ========\n{}\n".format(index + 1, text)
                    all_text.append(text)
                    index += 1
            if args.reverse:
                print("==== ORIGINAL REVERSED ====")
                print(textr)
            print(text)
            maketree(os.path.join(SAMPLE_DIR, args.run_name))
            gs = sess.run(tf.compat.v1.train.get_global_step())
            with open(
                os.path.join(SAMPLE_DIR, args.run_name, "samples-{}").format(gs),
                "w",
                encoding=args.encoding,
            ) as o:
                o.write("\n".join(all_text))

        def validation():
            print("Calculating validation loss...")
            losses = []
            for batch in tqdm.tqdm(val_batches):
                losses.append(sess.run(val_loss, feed_dict={val_context: batch}))
            v_val_loss = np.mean(losses)
            v_summary = sess.run(val_loss_summary, feed_dict={val_loss: v_val_loss})
            gs = sess.run(tf.compat.v1.train.get_global_step())
            summary_log.add_summary(v_summary, gs)
            summary_log.flush()
            print(
                "[{counter} | {time:2.2f}] validation loss = {loss:2.2f}".format(
                    counter=gs, time=time.time() - start_time, loss=v_val_loss
                )
            )

        def sample_batch():
            return [data_sampler.sample(hparams.n_ctx) for _ in range(args.batch_size)]

        def delete_previous_checkpoints():
            gs = sess.run(tf.compat.v1.train.get_global_step())
            for fname in os.listdir(f"checkpoint/{args.run_name}"):
                if regex.match(r"model-*", fname) and not regex.match(
                    r"model-" + str(gs) + r"*", fname
                ):
                    print(f"(deleting former checkpoint: {fname})")
                    os.remove(os.path.join("checkpoint", args.run_name, fname))

        avg_loss = (0.0, 0.0)
        start_time = time.time()

        try:
            while True:

                gs = sess.run(tf.compat.v1.train.get_global_step()) + 1

                if gs % args.save_every == 0:
                    try:
                        save()
                        delete_previous_checkpoints()
                    except:
                        print("\u001b[31mUNABLE TO SAVE, FREE UP SOME MEMORY?\u001b[0m")
                if gs % args.sample_every == 0:
                    generate_samples()
                if args.val_every > 0 and (gs % args.val_every == 0 or gs == 1):
                    validation()

                if args.accumulate_gradients > 1:
                    sess.run(opt_reset)
                    for _ in range(args.accumulate_gradients):
                        smpl_batch = sample_batch()
                        sess.run(opt_compute, feed_dict={context: smpl_batch})
                    (v_loss, v_summary) = sess.run((opt_apply, summaries))
                else:
                    smpl_batch = sample_batch()
                    (_, v_loss, v_summary) = sess.run(
                        (opt_apply, loss, summaries), feed_dict={context: smpl_batch},
                    )

                summary_log.add_summary(v_summary, gs)

                avg_loss = (avg_loss[0] * 0.9999 + v_loss, avg_loss[1] * 0.9999 + 1.0)

                msg = "[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f} lr={lr}".format(
                    counter=gs,
                    time=time.time() - start_time,
                    loss=v_loss,
                    avg=avg_loss[0] / avg_loss[1],
                    lr=sess.run(learning_rate),
                )

                if args.print_train_sample:
                    msg += " | Training on: "
                    # get tty width, https://stackoverflow.com/a/943921
                    columns = int(os.popen("stty size", "r").read().split()[1])
                    if args.reverse:
                        smpl = enc.decode(smpl_batch[0][::-1])
                    else:
                        smpl = enc.decode(smpl_batch[0])
                    msg += smpl.replace("\n", " ")[: columns - len(msg) - 5] + "..."
                print(msg)

        except KeyboardInterrupt:
            print("interrupted")
            save()
            delete_previous_checkpoints()


if __name__ == "__main__":
    main()
