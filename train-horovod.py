#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./train --dataset <file|directory|glob>

import tensorflow as tf
import numpy as np
import random
import signal
import time
import json
import fire
import os

import horovod.tensorflow as hvd

import model, sample, encoder
import encoder_sp as encoder_sp
import memory_saving_gradients as msv
from load_dataset import load_dataset, Sampler

CHECKPOINT_DIR = "checkpoint"
SAMPLE_DIR = "samples"

hvd.init()


def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass


def train_main(
    dataset,
    model_name="117M",
    seed=None,
    batch_size=1,
    sample_length=1023,
    sample_num=1,
    sample_every=4500,
    run_name="run1",
    restore_from="latest",
    save_every=2000,
    combine=50000,
    learning_rate=0.00002,
    optimizer="sgd",
    mixed_precision=False,
    memory_saving_gradients=True,
    only_train_transformer_layers=True,
    allreduce_on_cpu=False,
    encoder_type="default",
):

    if encoder_type == "default":
        enc = encoder.get_encoder(model_name)
    elif encoder_type == 'sentencepiece':
        enc = encoder_sp.get_encoder("models", model_name)
    hparams = model.default_hparams()
    with open(os.path.join("models", model_name, "hparams.json")) as f:
        hparams.override_from_dict(json.load(f))

    if sample_length is None:
        sample_length = hparams.n_ctx // 2
    elif sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx
        )

    # TF config

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = model.model(hparams=hparams, X=context)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output["logits"][:, :-1]
            )
        )

        tf_sample = sample.sample_sequence(
            hparams=hparams,
            length=sample_length,
            context=context,
            batch_size=batch_size,
            temperature=0.8,
            top_k=40,
        )

        all_vars = [v for v in tf.trainable_variables() if "model" in v.name]
        train_vars = (
            [v for v in all_vars if "/h" in v.name]
            if only_train_transformer_layers
            else all_vars
        )

        learning_rate = learning_rate * hvd.size()

        if optimizer == "adam":
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer == "sgd":
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        else:
            exit("Bad optimizer:", optimizer)

        if memory_saving_gradients:
            opt_grads = msv.gradients(loss, train_vars)
        else:
            opt_grads = tf.gradients(loss, train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads)

        # https://developer.nvidia.com/automatic-mixed-precision
        if mixed_precision:
            opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

        if hvd.rank() == 0:
            summary_loss = tf.summary.scalar("loss", loss)
            summary_lr = tf.summary.scalar("learning_rate", learning_rate)
            summaries = tf.summary.merge([summary_lr, summary_loss])
            summary_log = tf.summary.FileWriter(os.path.join(CHECKPOINT_DIR, run_name))

        # bottom of that pages:
        # https://github.com/horovod/horovod/blob/80167f6dea0ba6b853d790a3d3a342368811f0da/docs/gpus.rst
        if allreduce_on_cpu:
            opt = hvd.DistributedOptimizer(opt, device_dense="/cpu:0")
        else:
            opt = hvd.DistributedOptimizer(opt)
        train_op = opt.minimize(loss, var_list=train_vars)

        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        bcast = hvd.broadcast_global_variables(0)

        saver = tf.train.Saver(
            var_list=all_vars, max_to_keep=1, keep_checkpoint_every_n_hours=8
        )

        sess.run(tf.global_variables_initializer())

        if restore_from == "latest":
            ckpt = tf.train.latest_checkpoint(os.path.join(CHECKPOINT_DIR, run_name))
            if ckpt is None:
                # Get fresh GPT weights if new run.
                ckpt = tf.train.latest_checkpoint(os.path.join("models", model_name))
        elif restore_from == "fresh":
            ckpt = tf.train.latest_checkpoint(os.path.join("models", model_name))
        else:
            ckpt = tf.train.latest_checkpoint(restore_from)
        print(str(hvd.local_rank()), "Loading checkpoint", ckpt)
        saver.restore(sess, ckpt)

        bcast.run()

        print(str(hvd.local_rank()), "Loading dataset...")
        chunks = load_dataset(enc, dataset, combine)
        data_sampler = Sampler(chunks)
        print(str(hvd.local_rank()), "dataset has", data_sampler.total_size, "tokens")
        print(str(hvd.local_rank()), "Training...")

        counter = 1
        if os.path.exists(os.path.join(CHECKPOINT_DIR, run_name, "counter")):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(os.path.join(CHECKPOINT_DIR, run_name, "counter"), "r") as fp:
                counter = int(fp.read()) + 1

        def save():
            maketree(os.path.join(CHECKPOINT_DIR, run_name))
            print(
                "Saving",
                os.path.join(CHECKPOINT_DIR, run_name, "model-{}").format(counter),
            )
            saver.save(
                sess,
                os.path.join(CHECKPOINT_DIR, run_name, "model"),
                global_step=counter,
            )
            with open(os.path.join(CHECKPOINT_DIR, run_name, "counter"), "w") as fp:
                fp.write(str(counter) + "\n")

        # https://github.com/horovod/horovod/issues/1903#issuecomment-618721148
        def on_exit():
            if hvd.rank() == 0:
                print("interrupted")
                save()

        # assigning exit to SIGINT/SIGTERM
        signal.signal(signal.SIGINT, on_exit)
        signal.signal(signal.SIGTERM, on_exit)

        def generate_samples():
            context_tokens = data_sampler.sample(1)
            all_text = []
            index = 0
            while index < sample_num:
                out = sess.run(
                    tf_sample, feed_dict={context: batch_size * [context_tokens]}
                )
                for i in range(min(sample_num - index, batch_size)):
                    text = enc.decode(out[i])
                    text = "======== SAMPLE {} ========\n{}\n".format(index + 1, text)
                    all_text.append(text)
                    index += 1
            print(text)
            maketree(os.path.join(SAMPLE_DIR, run_name))
            with open(
                os.path.join(SAMPLE_DIR, run_name, "samples-{}").format(counter), "w"
            ) as fp:
                fp.write("\n".join(all_text))

        avg_loss = (0.0, 0.0)
        start_time = time.time()

        while True:

            batch = [data_sampler.sample(1024) for _ in range(batch_size)]

            if hvd.rank() == 0:
                (_, v_loss, v_summary) = sess.run(
                    (train_op, loss, summaries),
                    feed_dict={context: batch},
                )

                summary_log.add_summary(v_summary, counter)

                avg_loss = (avg_loss[0] * 0.99 + v_loss, avg_loss[1] * 0.99 + 1.0)

                counter += 1

                if counter % save_every == 0:
                    save()
                if counter % sample_every == 0:
                    generate_samples()

                print(
                    "[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}".format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1],
                    )
                )

            else:

                (_, v_loss) = sess.run(
                    (train_op, loss),
                    feed_dict={context: batch},
                )


if __name__ == "__main__":
    fire.Fire(train_main)
