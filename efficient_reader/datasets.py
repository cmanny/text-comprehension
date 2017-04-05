import requests
import tarfile
import os
import pickle
import re
from collections import Counter
import tensorflow as tf

class CBTExample(object):
    def __init__(self, context, query, answer):
        self.context = context
        self.query = query
        self.answer = answer

    def index_list(self, vocab_index):
        self.i_context = [vocab_index[token] for token in self.context]
        self.i_query = [vocab_index[token] for token in self.query]
        self.i_answer = [vocab_index[token] for token in self.answer]
        return (self.i_context, self.i_query, self.i_answer)


class CBTDataSet(object):
    _NAMED_ENTITY = {
        "train": "cbtest_NE_train.txt",
        "valid": "cbtest_NE_valid_2000ex.txt",
        "test": "cbtest_NE_test_2500ex.txt"
    }

    def __init__(self, data_dir="data", in_memory=False, name="cbt_data",
                 *args, **kwargs):
        self.in_memory = in_memory
        self.top_data_dir = data_dir
        self.name = name

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        self.from_url("http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz")

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
        with tarfile.open(name=file_name) as trf:
            directory = os.path.join(self.top_data_dir, self.name)
            os.mkdir(directory)
            trf.extractall(directory)

    @property
    def inner_data_dir(self):
        return os.path.join(self.inner_data, "CBTest", "data")

    @classmethod
    def clean(self, string):
        return [re.sub("[\n0-9]", '', x) for x in string.split(" ")]

    @classmethod
    def get_cqa_words(self, cqa):
        cqa_split = cqa.split("\n21 ")
        context = self.clean(cqa_split[0])
        last_line = cqa_split[1]
        query, answer = [self.clean(x) for x in last_line.split("\t", 2)[:2]]
        return context, query, answer

    @classmethod
    def tfrecord_example(ic, iq, ia):
        return tf.train.Example(
             features = tf.train.Features(
                 feature = {
                     'document': tf.train.Feature(
                         int64_list=tf.train.Int64List(value=ic)),
                     'query': tf.train.Feature(
                         int64_list=tf.train.Int64List(value=iq)),
                     'answer': tf.train.Feature(
                         int64_list=tf.train.Int64List(value=ia))
                     }
              )
        )

    def named_entities(self):
        print("[*] Creating records from Named Entities")
        self.obj = {
            "train": [],
            "valid": [],
            "test": [],
            "vocab": dict()
        }
        self.counter = Counter()
        for s, f_name in self._NAMED_ENTITY.items():
            full_path = os.path.join(self.inner_data_dir, f_name)
            with open(full_path, 'r') as f:
                file_string = f.read()
                for cqa in file_string.split("\n\n"):
                    if len(cqa) < 5:
                        break
                    context, query, answer = self.get_cqa_words(cqa)
                    print(s)
                    # self.obj[s].append(
                    #     CBTExample(context, query, answer)
                    # )
                    for token in context + query + answer:
                        self.counter[token] += 1
        # Get all words in counter, and create word-id mapping
        words, _ = zip(*self.counter.most_common())
        self.obj["vocab"] = {token: i for i, token in enumerate(words)}
        return self.obj

    def generate_tfrecord(self, name, examples, criterion=lambda x: True,
                          force=False):
        out_name = os.path.join("tfrecords", name + ".tfrecords")
        if os.path.exists(out_name) and not force:
            return out_name
        writer = tf.python_io.TFRecordWriter(out_name)
        for cbt_example in examples:
            if not criterion(cbt_example):
                continue
            ic, iq, ia = cbt_example.index_list(self.obj["vocab"])
            example = tf.train.Example(
                 features = tf.train.Features(
                     feature = {
                         'document': tf.train.Feature(
                             int64_list=tf.train.Int64List(value=ic)),
                         'query': tf.train.Feature(
                             int64_list=tf.train.Int64List(value=iq)),
                         'answer': tf.train.Feature(
                             int64_list=tf.train.Int64List(value=ia))
                         }
                  )
            )
            serialized = example.SerializeToString()
            writer.write(serialized)
        return out_name
