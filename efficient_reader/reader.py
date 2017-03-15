import os
import pickle
import re
from collections import Counter
import tensorflow as tf

def clean(string):
    return [re.sub("[\n0-9]", '', x) for x in string.split(" ")]


def get_cqa_words(cqa):
    cqa_split = cqa.split("\n21 ")
    context = clean(cqa_split[0])
    last_line = cqa_split[1]
    query, answer = [clean(x) for x in last_line.split("\t", 2)[:2]]
    return context, query, answer

def counts():
    cache = 'counter.pickle'
    if os.path.exists(cache):
        with open(cache, 'r') as f:
            return pickle.load(f)

    directories = ['data/train/', 'data/valid/', 'data/test/']
    files = [directory + file_name for directory in directories for file_name in os.listdir(directory)]
    counter = Counter()
    for file_name in files:
        with open(file_name, 'r') as f:
            file_string = f.read()
            for cqa in file_string.split("\n\n"):
                    if len(cqa) < 5:
                        break
                    context, query, answer = get_cqa_words(cqa)
                    for token in context + query + answer:
                        counter[token] += 1
    with open(cache, 'w') as f:
        pickle.dump(counter, f)

    return counter

def tokenize(index, word):
    directories = ['data/train/', 'data/valid/', 'data/test/']
    for directory in directories:
        out_name = directory.split('/')[-2] + '.tfrecords'
        writer = tf.python_io.TFRecordWriter(out_name)
        files = map(lambda file_name: directory + file_name, os.listdir(directory))
        for file_name in files:
            with open(file_name, 'r') as f:
                file_string = f.read()
                for cqa in file_string.split("\n\n"):
                    if len(cqa) < 5:
                        break
                    context, query, answer = get_cqa_words(cqa)
                    context = [index[token] for token in context]
                    query = [index[token] for token in query]
                    answer = [index[token] for token in answer]
                    example = tf.train.Example(
                         features = tf.train.Features(
                             feature = {
                                 'document': tf.train.Feature(
                                     int64_list=tf.train.Int64List(value=context)),
                                 'query': tf.train.Feature(
                                     int64_list=tf.train.Int64List(value=query)),
                                 'answer': tf.train.Feature(
                                     int64_list=tf.train.Int64List(value=answer))
                                 }
                          )
                    )
                    serialized = example.SerializeToString()
                    writer.write(serialized)

def main():
    counter = counts()
    print('num words',len(counter))
    word, _ = zip(*counter.most_common())
    index = {token: i for i, token in enumerate(word)}
    tokenize(index, word)
    print('DONE')

if __name__ == "__main__":
    main()
