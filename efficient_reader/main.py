import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import models
from datasets import CBTDataSet, Sampler


parser = argparse.ArgumentParser(description='How to run this')
parser.add_argument(
    "-epoch",
    type=int,
    default=3,
    help="Epochs to train [3]"
)
parser.add_argument(
    "-learning_rate",
    type=float,
    default=5e-5,
    help="Learning rate [0.00005]"
)
parser.add_argument(
    "-data_dir",
    type=str,
    default="data",
    help="The name of data directory [data]"
)
parser.add_argument(
    "-dataset",
    type=str,
    default="cbt",
    help="The name of dataset [cbt]"
)
parser.add_argument(
    "-checkpoints",
    type=str,
    default="checkpoints",
    help="Directory name to save the checkpoints [checkpoints]"
)
parser.add_argument(
    "-forward_only",
    type=bool,
    default=False,
    help="True for forward only, False for training [False]"
)
parser.add_argument(
    "-sample",
    type=str,
    default="full_train",
    help="Dataset sample to use"
)
parser.add_argument(
    "-model",
    type=str,
    default="full_train",
    help="Model to use as previously saved"
)
args = parser.parse_args()

def main():
    cbt_dataset = CBTDataSet(data_dir="data")
    sampler_dict = {
        "train": [
            Sampler("full_train", lambda x: True),
            Sampler("word_distance_pass", models.word_distance),
            Sampler("word_distance_fail", lambda x: not models.word_distance(x)),
            Sampler("frequency_pass", models.frequency),
            Sampler("frequency_fail", lambda x: not models.frequency(x))
        ],
        "valid": [
            Sampler("full_valid", lambda x: True)
        ],
        "test": [
            Sampler("full_test", lambda x: True)
        ]
    }
    cbt_dataset.named_entities(sampler_dict)

    print("running model {}".format(args.sample))
    ASReader().run(args.sample, args.model, args.forward_only)


if __name__ == '__main__':
    main()
