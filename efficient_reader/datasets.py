import os
import requests
import tarfile


class CBTDataSet(object):
    def __init__(self, data_dir, in_memory=False, name="dts", *args, **kwargs):
        self.in_memory = in_memory
        self.top_data_dir = data_dir
        self.name = name

    # Function to automatically handle data
    def auto_setup(self):
        self.from_url("http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz")
        self.get_ne_data()

    def from_url(self, url):
        self.inner_data = os.path.join(self.top_data_dir, self.name)
        if os.path.exists(self.inner_data):
            return
        file_name = os.path.join(self.top_data_dir, self.name + ".tar.gz")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:10.0) \
                               Gecko/20100101 Firefox/10.0',
            }
            r = requests.get(url, headers=headers)
            with open(file_name, 'wb') as outfile:
                outfile.write(r.content)
        except IOError as e:
            print("Could not get the file from URL: " + url)
            raise
        with tarfile.open(name=file_name) as tf:
            directory = os.path.join(self.top_data_dir, self.name)
            os.mkdir(directory)
            tf.extractall(directory)

    @property
    def data_dir(self):
        return os.path.join(self.inner_data, "CBTest", "data")

    # I really shoudn't do this
    def get_ne_data(self):
        with open(os.path.join(self.data_dir, "cbtest_NE_test_2500ex.txt")) as test:
            self.test = test.readlines()
            print(self.test[0])

        with open(os.path.join(self.data_dir, "cbtest_NE_train.txt")) as train:
            self.train = train.readlines()

        with open(os.path.join(self.data_dir, "cbtest_NE_valid_2000ex.txt")) as valid:
            self.valid = valid.readlines()

import os
import pickle
from collections import Counter
import tensorflow as tf

def get_cqa_words(cqa):
    cqa_split = cqa.split("\n21 ")
    context = cqa_split[0].split(" ")
    last_line = cqa_split[1]
    query, answer = [x.split(" ") for x in last_line.split(" ", 2)[:2]]
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
