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

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('vocab_size', 62255, 'Vocabulary size')
flags.DEFINE_integer('embedding_size', 384, 'Embedding dimension')
flags.DEFINE_integer('hidden_size', 384, 'Hidden units')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('epochs', 2, 'Number of epochs to train/test')
flags.DEFINE_boolean('training', True, 'Training or testing a model')
flags.DEFINE_string('name', '', 'Model name (used for statistics and model path')
flags.DEFINE_float('dropout_keep_prob', 0.9, 'Keep prob for embedding dropout')
flags.DEFINE_float('l2_reg', 0.0001, 'l2 regularization for embeddings')

def read_records(index=0, sample_name="full_train"):
  train_queue = tf.train.string_input_producer(['tfrecords/' + sample_name + '.tfrecords'], num_epochs=FLAGS.epochs)
  validation_queue = tf.train.string_input_producer(['tfrecords/full_valid.tfrecords'], num_epochs=FLAGS.epochs)
  test_queue = tf.train.string_input_producer(['tfrecords/full_test.tfrecords'], num_epochs=FLAGS.epochs)
  print(train_queue)

  queue = tf.QueueBase.from_list(index, [train_queue, validation_queue, test_queue])
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
        'document': tf.VarLenFeature(tf.int64),
        'query': tf.VarLenFeature(tf.int64),
        'answer': tf.FixedLenFeature([], tf.int64)
      })
  document = sparse_ops.serialize_sparse(features['document'])
  query = sparse_ops.serialize_sparse(features['query'])
  answer = features['answer']

  document_batch_serialized, query_batch_serialized, answer_batch = tf.train.shuffle_batch(
      [document, query, answer], batch_size=FLAGS.batch_size,
      capacity=2000,
      min_after_dequeue=1000)

  sparse_document_batch = sparse_ops.deserialize_many_sparse(document_batch_serialized, dtype=tf.int64)
  sparse_query_batch = sparse_ops.deserialize_many_sparse(query_batch_serialized, dtype=tf.int64)

  document_batch = tf.sparse_tensor_to_dense(sparse_document_batch)
  document_weights = tf.sparse_to_dense(sparse_document_batch.indices, sparse_document_batch.dense_shape, 1)

  query_batch = tf.sparse_tensor_to_dense(sparse_query_batch)
  query_weights = tf.sparse_to_dense(sparse_query_batch.indices, sparse_query_batch.dense_shape, 1)

  return document_batch, document_weights, query_batch, query_weights, answer_batch

def inference(documents, doc_mask, query, query_mask):

  # the embedding may already exist so we use tf.get_variable to find it if
  # it already exists, initialize to random uniform values if created
  embedding = tf.get_variable('embedding',
              [FLAGS.vocab_size, FLAGS.embedding_size],
              initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05))

  # using regularizer to reduce overfitting
  regularizer = tf.nn.l2_loss(embedding)

  # look up embeddings for document, and use dropout
  doc_emb = tf.nn.dropout(tf.nn.embedding_lookup(embedding, documents), FLAGS.dropout_keep_prob)
  doc_emb.set_shape([None, None, FLAGS.embedding_size])

  # look up embeddings for query, and use dropout
  query_emb = tf.nn.dropout(tf.nn.embedding_lookup(embedding, query), FLAGS.dropout_keep_prob)
  query_emb.set_shape([None, None, FLAGS.embedding_size])

  with tf.variable_scope('document', initializer=orthogonal_initializer()):
    # we use GRU cells as given in the paper, however other cells could be used
    fwd_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_size)
    back_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_size)

    # we need to specify the query length as an rnn argument
    doc_len = tf.reduce_sum(doc_mask, axis=1)

    # one may either use output_states, the first return value, or
    # final_state, the second return value. The first return value gives
    # all of the outputs after every input, while the second return value
    # gives only the final output aftre every input has been passed through
    output_states, _ = tf.nn.bidirectional_dynamic_rnn(
        fwd_cell, back_cell, doc_emb, sequence_length=tf.to_int64(doc_len), dtype=tf.float32
    )

    # concatenate forward and backward cells into one tensor
    h_doc = tf.concat(axis=2, values=output_states)

  with tf.variable_scope('query', initializer=orthogonal_initializer()):
    # we use GRU cells as given in the paper, however other cells could be used
    fwd_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_size)
    back_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_size)

    # we need to specify the query length as an rnn argument
    query_len = tf.reduce_sum(query_mask, axis=1)
    output_states, _ = tf.nn.bidirectional_dynamic_rnn(
        fwd_cell, back_cell, query_emb, sequence_length=tf.to_int64(query_len), dtype=tf.float32
    )

    # concatenate forward and backward cells into one tensor
    h_query = tf.concat(axis=2, values=output_states)

  M = tf.matmul(h_doc, h_query, adjoint_b=True)
  M_mask = tf.to_float(tf.matmul(tf.expand_dims(doc_mask, -1), tf.expand_dims(query_mask, 1)))

  alpha = softmax(M, 1, M_mask)
  beta = softmax(M, 2, M_mask)

  #query_importance = tf.expand_dims(tf.reduce_mean(beta, reduction_indices=1), -1)
  query_importance = tf.expand_dims(tf.reduce_sum(beta, 1) / tf.to_float(tf.expand_dims(doc_len, -1)), -1)

  s = tf.squeeze(tf.matmul(alpha, query_importance), [2])

  unpacked_s = zip(tf.unstack(s, FLAGS.batch_size), tf.unstack(documents, FLAGS.batch_size))

  # create the vocabulary x batch sized list votes for words
  y_hat = tf.stack([tf.unsorted_segment_sum(attentions, sentence_ids, FLAGS.vocab_size) for (attentions, sentence_ids) in unpacked_s])

  return y_hat, regularizer

def train(y_hat, regularizer, document, doc_weight, answer):

  index = tf.range(0, FLAGS.batch_size) * FLAGS.vocab_size + tf.to_int32(answer)
  flat = tf.reshape(y_hat, [-1])
  relevant = tf.gather(flat, index)
  relevant = tf.check_numerics(relevant, "relevant")

  log = tf.check_numerics(tf.log(relevant + 1e-10), "log")

  loss = -tf.reduce_mean(log) + FLAGS.l2_reg * regularizer
  loss = tf.check_numerics(loss, "loss")

  global_step = tf.Variable(0, name="global_step", trainable=False)

  accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(y_hat, 1), answer)))

  optimizer = tf.train.AdamOptimizer()
  grads_and_vars = optimizer.compute_gradients(loss)

  # use clipping on gradients to prevent explosion or implosion
  capped_grads_and_vars = [(tf.clip_by_value(grad, -5, 5), var) for (grad, var) in grads_and_vars]
  train_op = optimizer.apply_gradients(capped_grads_and_vars, global_step=global_step)

  tf.summary.scalar('loss', loss)
  tf.summary.scalar('accuracy', accuracy)
  return loss, train_op, global_step, accuracy

def main(sample_name, model_name, forward_only):
  model_path = 'models/' + model_name + "_l2reg_" + str(FLAGS.l2_reg)
  if not os.path.exists(model_path):
      os.makedirs(model_path)
  dataset = tf.placeholder_with_default(2 if forward_only else 0, [])
  document_batch, document_weights, query_batch, query_weights, answer_batch = read_records(index=dataset, sample_name=sample_name)

  y_hat, reg = inference(document_batch, document_weights, query_batch, query_weights)
  loss, train_op, global_step, accuracy = train(y_hat, reg, document_batch, document_weights, answer_batch)
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
    if not FLAGS.training:
      saver_variables = filter(lambda var: var.name != 'input_producer/limit_epochs/epochs:0', saver_variables)
      saver_variables = filter(lambda var: var.name != 'smooth_acc:0', saver_variables)
      saver_variables = filter(lambda var: var.name != 'avg_acc:0', saver_variables)
    saver = tf.train.Saver(saver_variables, max_to_keep=100)

    sess.run([
        tf.global_variables_initializer(),
        tf.local_variables_initializer()])
    model = tf.train.latest_checkpoint(model_path)
    if cp != None:
        model = model_path + "/" + cp
    if model:
      print('Restoring ' + model)
      saver.restore(sess, model)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    start_time = time.time()
    accumulated_accuracy = 0
    try:
      if not forward_only:
        while not coord.should_stop():
          loss_t, _, step, acc = sess.run([loss, train_op, global_step, accuracy], feed_dict={dataset: 0})
          elapsed_time, start_time = time.time() - start_time, time.time()
          print(step, loss_t, acc, elapsed_time)
          if step % 5 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
          if step % 100 == 0:
            saver.save(sess, model_path + "/run", global_step=step)
      else:
        step = 0
        ta_path = "ta_{}.pickle".format(model_name)
        while not coord.should_stop():
          acc = sess.run(accuracy, feed_dict={dataset: 2})
          step += 1
          accumulated_accuracy += (acc - accumulated_accuracy) / step
          elapsed_time, start_time = time.time() - start_time, time.time()
          print(accumulated_accuracy, acc, elapsed_time)

          # stop test after a few runs
          if step % 20 == 0:
            if os.path.exists(ta_path):
              with open(ta_path, 'r+') as tap:
                ta_dict = pickle.load(tap)
                ta_dict[model_name] += [accumulated_accuracy]
                pickle.dump(ta_dict, tap)
            else:
              with open(ta_path, 'w') as tap:
                ta_dict = dict()
                ta_dict[model_name] = [accumulated_accuracy]
                pickle.dump(ta_dict, tap)
            coord.request_stop()
    except tf.errors.OutOfRangeError:
      print('Done!')
    finally:
      coord.request_stop()
    coord.join(threads)


if __name__ == "__main__":
  main()
