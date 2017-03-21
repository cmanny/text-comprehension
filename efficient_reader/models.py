import os
from glob import glob
import time
import numpy as np
import tensorflow as tf

from cells import LSTMCell, MultiRNNCellWithSkipConn
from utils import load_vocab, load_dataset, array_pad

class BaseModel(object):
    def __init__(self):
        self.vocab = None
        self.data = None

    def save(self, sess, checkpoint_dir, dataset_name, global_step=None):
        self.saver = tf.train.Saver()

        print(" [*] Saving checkpoints...")
        model_name = type(self).__name__ or "Reader"
        if self.batch_size:
            model_dir = "%s_%s_%s" % (model_name, dataset_name, self.batch_size)
        else:
            model_dir = dataset_name

        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess,
                os.path.join(checkpoint_dir, model_name), global_step=global_step)

    def load(self, sess, checkpoint_dir, dataset_name):
        model_name = type(self).__name__ or "Reader"
        self.saver = tf.train.Saver()

        print(" [*] Loading checkpoints...")
        if self.batch_size:
            model_dir = "%s_%s_%s" % (model_name, dataset_name, self.batch_size)
        else:
            model_dir = dataset_name
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

class DeepLSTM(BaseModel):
    """Deep LSTM model."""
    def __init__(self, size=256, depth=3, batch_size=32,
                             keep_prob=0.1, max_nsteps=1000,
                             checkpoint_dir="checkpoint", forward_only=False):
        super(DeepLSTM, self).__init__()

        self.size = int(size)
        self.depth = int(depth)
        self.batch_size = int(batch_size)
        self.output_size = self.depth * self.size
        self.keep_prob = float(keep_prob)
        self.max_nsteps = int(max_nsteps)
        self.checkpoint_dir = checkpoint_dir

        start = time.clock()
        print(" [*] Building Deep LSTM...")
        self.cell = LSTMCell(size, forget_bias=0.0)
        # if not forward_only and self.keep_prob < 1:
        #     self.cell = tf.contrib.rnn.DropoutWrapper(self.cell, output_keep_prob=keep_prob)
        self.stacked_cell = MultiRNNCellWithSkipConn([self.cell] * depth)

        self.initial_state = self.stacked_cell.zero_state(batch_size, tf.float32)

    def prepare_model(self, data_dir, dataset_name, vocab_size):
        if not self.vocab:
            self.vocab, self.rev_vocab = load_vocab(data_dir, dataset_name, vocab_size)
            print(" [*] Loading vocab finished.")

        self.vocab_size = len(self.vocab)

        self.emb = tf.get_variable("emb", [self.vocab_size, self.size])

        # inputs
        self.inputs = tf.placeholder(tf.int32, [self.batch_size, self.max_nsteps])
        embed_inputs = tf.nn.embedding_lookup(self.emb, tf.transpose(self.inputs))

        tf.histogram_summary("embed", self.emb)

        # output states
        _, states = tf.nn.dynamic_rnn(self.stacked_cell,
                                      tf.unpack(embed_inputs),
                                      dtype=tf.float32,
                                      initial_state=self.initial_state)
        self.batch_states = tf.pack(states)

        self.nstarts = tf.placeholder(tf.int32, [self.batch_size, 3])
        outputs = tf.pack([tf.slice(self.batch_states, nstarts, [1, 1, self.output_size])
                for idx, nstarts in enumerate(tf.unpack(self.nstarts))])

        self.outputs = tf.reshape(outputs, [self.batch_size, self.output_size])

        self.W = tf.get_variable("W", [self.vocab_size, self.output_size])
        tf.histogram_summary("weights", self.W)
        tf.histogram_summary("output", outputs)

        self.y = tf.placeholder(tf.float32, [self.batch_size, self.vocab_size])
        self.y_ = tf.matmul(self.outputs, self.W, transpose_b=True)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(self.y_, self.y)
        tf.scalar_summary("loss", tf.reduce_mean(self.loss))

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.scalar_summary("accuracy", self.accuracy)

        print(" [*] Preparing model finished.")

    def train(self, sess, vocab_size, epoch=25, learning_rate=0.0002,
                        momentum=0.9, decay=0.95, data_dir="data", dataset_name="cbt"):
        self.prepare_model(data_dir, dataset_name, vocab_size)

        start = time.clock()
        print(" [*] Calculating gradient and loss...")
        self.optim = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(self.loss)
        print(" [*] Gradient and loss finished after %.2fs" % (time.clock() - start))

        sess.run(tf.initialize_all_variables())

        if self.load(sess, self.checkpoint_dir, dataset_name):
            print(" [*] Deep LSTM checkpoint is loaded.")
        else:
            print(" [*] There is no checkpoint for this model.")

        y = np.zeros([self.batch_size, self.vocab_size])

        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("/tmp/deep", sess.graph_def)

        counter = 0
        start_time = time.time()
        for epoch_idx in xrange(epoch):
            data_loader = load_dataset(data_dir, dataset_name, vocab_size)

            batch_stop = False
            while True:
                y.fill(0)
                inputs, nstarts, answers = [], [], []
                batch_idx = 0
                while True:
                    try:
                        (_, document, question, answer, _), data_idx, data_max_idx = data_loader.next()
                    except StopIteration:
                        batch_stop = True
                        break

                    # [0] means splitter between d and q
                    data = [int(d) for d in document.split()] + [0] + \
                            [int(q) for q in question.split() for q in question.split()]

                    if len(data) > self.max_nsteps:
                        continue

                    inputs.append(data)
                    nstarts.append(len(inputs[-1]) - 1)
                    y[batch_idx][int(answer)] = 1

                    batch_idx += 1
                    if batch_idx == self.batch_size: break
                if batch_stop: break

                FORCE=False
                if FORCE:
                    inputs = array_pad(inputs, self.max_nsteps, pad=-1, force=FORCE)
                    nstarts = np.where(inputs==-1)[1]
                    inputs[inputs==-1]=0
                else:
                    inputs = array_pad(inputs, self.max_nsteps, pad=0)
                nstarts = [[nstart, idx, 0] for idx, nstart in enumerate(nstarts)]

                _, summary_str, cost, accuracy = sess.run(
                    [self.optim, merged, self.loss, self.accuracy],
                    feed_dict={
                        self.inputs: inputs,
                        self.nstarts: nstarts,
                        self.y: y
                    }
                )
                if counter % 10 == 0:
                    writer.add_summary(summary_str, counter)
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, accuracy: %.8f" \
                            % (epoch_idx, data_idx, data_max_idx, time.time() - start_time, np.mean(cost), accuracy))
                counter += 1
            self.save(sess, self.checkpoint_dir, dataset_name)

    def test(self, voab_size):
        self.prepare_model(data_dir, dataset_name, vocab_size)
