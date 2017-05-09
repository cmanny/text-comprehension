import os
import pickle
import time
import random
import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.python.ops import sparse_ops
from network_util import softmax, orthogonal_initializer
from tensorflow.contrib.tensorboard.plugins import projector

def word_distance(example):
    i_context, i_query, i_answer, i_candidates = example.index_list()
    i_missing = example.vocab["XXXXX"]
    missing_index = i_query.index(i_missing)
    positions = {word: [] for word in i_context}
    for i, word in enumerate(i_context):
        positions[word].append(i)
    penalties = [0 for _ in i_candidates]
    for i, candidate in enumerate(i_candidates):
        for position in positions[candidate]:
            for j, word in enumerate(i_query):
                context_index = position + j - missing_index
                if word in positions:
                    distances = [abs(context_index - x) for x in positions[word]]
                else:
                    distances = [5]
                penalty = min(5, *distances)
                penalties[i] += penalty
    predicted = i_candidates[penalties.index(min(penalties))]
    return predicted == i_answer[0]

def frequency(example):
    i_context, i_query, i_answer, i_candidates = example.index_list()
    counter = Counter()
    for word in i_context:
        counter[word] += 1
    predicted = max(i_candidates, key=(lambda word: counter[word]))
    return predicted == i_answer[0]

class ASReader(object):
    def __init__(self):
        self.dropout_keep_prob = 0.9
        self.hidden_size = 384
        self.embedding_size = 384
        self.vocab_size = 62255
        self.batch_size = 32
        self.epochs = 2
        self.l2_reg_lambda = 0.0001

    def _cg_read(self, set_name):
        set_queue = tf.train.string_input_producer(['tfrecords/' + set_name + '.tfrecords'], num_epochs=self.epochs)
        queue = tf.QueueBase.from_list(0, [set_queue])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'document': tf.VarLenFeature(tf.int64),
                'query': tf.VarLenFeature(tf.int64),
                'answer': tf.FixedLenFeature([], tf.int64)
        })
        context = sparse_ops.serialize_sparse(features['document'])
        query = sparse_ops.serialize_sparse(features['query'])
        answer = features['answer']

        context_bs, query_bs, answer_bs = tf.train.shuffle_batch(
            [context, query, answer], batch_size=self.batch_size,
            capacity=2000,
            min_after_dequeue=1000
        )

        sparse_context_batch = sparse_ops.deserialize_many_sparse(context_bs, dtype=tf.int64)
        sparse_query_batch = sparse_ops.deserialize_many_sparse(query_bs, dtype=tf.int64)

        self.context_batch = tf.sparse_tensor_to_dense(sparse_context_batch)
        self.context_weights = tf.sparse_to_dense(sparse_context_batch.indices, sparse_context_batch.dense_shape, 1)

        self.query_batch = tf.sparse_tensor_to_dense(sparse_query_batch)
        self.query_weights = tf.sparse_to_dense(sparse_query_batch.indices, sparse_query_batch.dense_shape, 1)

        self.answer_batch = answer_bs

    def _cg_inference(self):

        # the embedding may already exist so we use tf.get_variable to find it if
        # it already exists, initialize to random uniform values if created
        embedding = tf.get_variable('embedding',
            [self.vocab_size, self.embedding_size],
            initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05)
        )

        # using regularizer to reduce overfitting
        self.regularizer = tf.nn.l2_loss(embedding)

        # look up embeddings for document, and use dropout
        context_emb = tf.nn.dropout(tf.nn.embedding_lookup(embedding, self.context_batch), self.dropout_keep_prob)
        context_emb.set_shape([None, None, self.embedding_size])

        # look up embeddings for query, and use dropout
        query_emb = tf.nn.dropout(tf.nn.embedding_lookup(embedding, self.query_batch), self.dropout_keep_prob)
        query_emb.set_shape([None, None, self.embedding_size])

        with tf.variable_scope('context', initializer=orthogonal_initializer()):
            # we need to specify the query length as an rnn argument
            context_len = tf.reduce_sum(self.context_weights, axis=1)

            # one may either use output_states, the first return value, or
            # final_state, the second return value. The first return value gives
            # all of the outputs after every input, while the second return value
            # gives only the final output aftre every input has been passed through
            output_states, _ = tf.nn.bidirectional_dynamic_rnn(
                tf.contrib.rnn.GRUCell(self.hidden_size),
                tf.contrib.rnn.GRUCell(self.hidden_size),
                context_emb,
                sequence_length=tf.to_int64(context_len),
                dtype=tf.float32
            )

            # concatenate forward and backward cells into one tensor
            h_context = tf.concat(axis=2, values=output_states)

        with tf.variable_scope('query', initializer=orthogonal_initializer()):
            # we need to specify the query length as an rnn argument
            query_len = tf.reduce_sum(self.query_weights, axis=1)
            output_states, _ = tf.nn.bidirectional_dynamic_rnn(
                tf.contrib.rnn.GRUCell(self.hidden_size),
                tf.contrib.rnn.GRUCell(self.hidden_size),
                query_emb,
                sequence_length=tf.to_int64(query_len),
                dtype=tf.float32
            )

            # concatenate forward and backward cells into one tensor
            h_query = tf.concat(axis=2, values=output_states)

        with tf.name_scope("attention"):
            M = tf.matmul(h_context, h_query, adjoint_b=True)
            M_mask = tf.to_float(tf.matmul(tf.expand_dims(self.context_weights, -1), tf.expand_dims(self.query_weights, 1)))

            alpha = softmax(M, 1, M_mask)
            beta = softmax(M, 2, M_mask)

            #query_importance = tf.expand_dims(tf.reduce_mean(beta, reduction_indices=1), -1)
            query_importance = tf.expand_dims(tf.reduce_sum(beta, 1) / tf.to_float(tf.expand_dims(context_len, -1)), -1)

            s = tf.squeeze(tf.matmul(alpha, query_importance), [2])

            unpacked_s = zip(tf.unstack(s, self.batch_size), tf.unstack(self.context_batch, self.batch_size))

            # create the vocabulary x batch sized list votes for words
        self.y_hat = tf.stack([tf.unsorted_segment_sum(attentions, sentence_ids, self.vocab_size) for (attentions, sentence_ids) in unpacked_s])

    def _cg_train(self):

        with tf.name_scope("loss"):
            index = tf.range(0, self.batch_size) * self.vocab_size + tf.to_int32(self.answer_batch)
            flat = tf.reshape(self.y_hat, [-1])
            relevant = tf.gather(flat, index)
            relevant = tf.check_numerics(relevant, "relevant")

            log = tf.check_numerics(tf.log(relevant + 1e-10), "log")

            loss = -tf.reduce_mean(log) + self.l2_reg_lambda * self.regularizer
            self.loss = tf.check_numerics(loss, "loss")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(self.y_hat, 1), self.answer_batch)))

        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer()
            grads_and_vars = optimizer.compute_gradients(loss)

            # use clipping on gradients to prevent explosion or implosion
            capped_grads_and_vars = [(tf.clip_by_value(grad, -5, 5), var) for (grad, var) in grads_and_vars]
            self.train_op = optimizer.apply_gradients(capped_grads_and_vars, global_step=self.global_step)

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)

    def run(self, sample_name, model_name, forward_only):
        model_path = 'models/' + model_name
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Build computation graph, first read samples, then infer, then optimize
        self._cg_read("full_test" if forward_only else sample_name)
        self._cg_inference()
        self._cg_train()

        # Display histograms of all trainable variables
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        summary_op = tf.summary.merge_all()

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(model_path, sess.graph)
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = "embedding"
            embedding.metadata_path = 'cache/embedding.meta'
            projector.visualize_embeddings(summary_writer, config)

            saver_variables = tf.global_variables()
            saver = tf.train.Saver(saver_variables, max_to_keep=100)
            sess.run([
                tf.global_variables_initializer(),
                tf.local_variables_initializer()]
            )

            model = tf.train.latest_checkpoint(model_path)
            if model:
              saver.restore(sess, model)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            start_time = time.time()
            accumulated_accuracy = 0

            # The
            try:
                if not forward_only:
                    while not coord.should_stop():

                        loss_t, _, step, acc = sess.run(
                          [self.loss, self.train_op, self.global_step, self.accuracy]
                        )
                        elapsed_time, start_time = time.time() - start_time, time.time()

                        if step % 10 == 0:
                            summary_str = sess.run(summary_op)
                            summary_writer.add_summary(summary_str, step)
                        if step % 100 == 0:
                            saver.save(sess, model_path + "/run", global_step=step)
                else:
                    while not coord.should_stop():
                        acc = sess.run(accuracy)
                        print(acc)
            except tf.errors.OutOfRangeError:
                print('Done!')
            finally:
                coord.request_stop()
            coord.join(threads)
