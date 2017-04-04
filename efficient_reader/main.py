import os
import argparse
import numpy as np
import tensorflow as tf
from datasets import CBTDataSet


parser = argparse.ArgumentParser(description='How to run this')
parser.add_argument(
    "-epoch",
    type=int,
    default=3,
    help="Epochs to train [3]"
)
# parser.add_argument(
#     "-vocab_size",
#     type=int,
#     default=10000,
#     help="The size of vocabulary [10000]"
# )
# Will be handled by the reader component
parser.add_argument(
    "-learning_rate",
    type=float,
    default=5e-5,
    help="Learning rate [0.00005]"
)
parser.add_argument(
    "-model",
    type=str,
    default="LSTM",
    help="The type of model to train and test [LSTM]"
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
args = parser.parse_args()

# model_dict = {
#     'ASReader': ASReader,
# }

def main():
    cbt_dataset = CBTDataSet(data_dir="data")
    obj = cbt_dataset.named_entities()
    train_tfrecord = cbt_dataset.generate_tfrecord("train_full", obj["train"])
    test_tfrecord = cbt_dataset.generate_tfrecord("test_full", obj["test"])
    valid_tfrecord = cbt_dataset.generate_tfrecord("valid_full", obj["valid"])

if __name__ == '__main__':
    main()
