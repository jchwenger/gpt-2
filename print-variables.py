import os
import json
import model
import argparse
import tensorflow as tf
# from tensorflow.core.protobuf import rewriter_config_pb2

parser = argparse.ArgumentParser(
    description="Print the list of variables (layers, etc.) of the GPT-2 model.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--model_name",
    type=str,
    default="117M",
    help="The name of the model. Defaults to '117M'"
)

def main(args):
    # get params
    hparams = model.default_hparams()
    with open(os.path.join("models", args.model_name, "hparams.json")) as f:
        hparams.override_from_dict(json.load(f))
    config = tf.compat.v1.ConfigProto()
    with tf.compat.v1.Session(config=config) as sess:
        context = tf.compat.v1.placeholder(tf.int32, [1, None])
        output = model.model(hparams=hparams, X=context)
        all_vars = [v for v in tf.compat.v1.trainable_variables() if "model" in v.name]
        name_pad = max([len(v.name) for v in all_vars])
        var_pad =  max([len(str(v.shape)) for v in all_vars])
        with open('gpt2-vars.txt', 'w') as o:
            for var in all_vars:
                msg = f"{var.name:{name_pad}} | shape: {str(var.shape):{var_pad}} | dtype: {var.dtype.name}"
                o.write(msg + "\n")
                print(msg)
        print("-" * 40)
        print("printed all variables to 'gpt2-vars.txt'")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
