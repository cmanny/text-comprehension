import os
import argparse
import numpy as np
import tensorflow as tf
from models import DeepLSTM
from utils import pp
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

model_dict = {
    'LSTM': DeepLSTM,
}

def main():
    cbt_dataset = CBTDataSet(data_dir="raw_data")
    pass
    # rewrite
    # tf.logging.set_verbosity(tf.logging.ERROR)
    # data_set = CBTDataSet("data", name="cbt_data")
    # data_set.auto_setup()
    # print(data_set.data_dir)
    # if not os.path.exists(args.checkpoints):
    #     print(" [*] Creating checkpoint directory...")
    #     os.makedirs(args.checkpoint_dir)
    #
    # with tf.device('/cpu:0'), tf.Session() as sess:
    #     model = model_dict[args.model](batch_size=32,
    #     checkpoint_dir=args.checkpoints, forward_only=args.forward_only)
    #
    #     if not args.forward_only:
    #         model.train(sess, args.vocab_size, args.epoch,
    #                     args.learning_rate, .9, .95,
    #                     args.data_dir, data_set.data_dir)
    #     else:
    #         model.load(sess, args.checkpoints, args.dataset)

if __name__ == '__main__':
    main()
