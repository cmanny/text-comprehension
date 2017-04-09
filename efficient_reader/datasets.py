import requests
import tarfile
import os
import pickle
import re
from collections import Counter
import tensorflow as tf

class Sampler(object):
    def __init__(self, name, filter_func):
        self.name = name
        self.filter_func = filter_func

        out_name = os.path.join("tfrecords/", name + ".tfrecords")
        self.writer = tf.python_io.TFRecordWriter(out_name)

    @classmethod
    def tfrecord_example(self, ic, iq, ia):
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

    def __call__(self, vocab, example):
        if self.filter_func(example):
            i_cqac = example.index_list(vocab)
            example = self.tfrecord_example(*i_cqac[:-1])
            print(example)
            serialized = example.SerializeToString()
            self.writer.write(serialized)



class CBTExample(object):
    def __init__(self, index, context, query, answer, candidates):
        self.index = index
        self.context = context
        self.query = query
        self.answer = answer
        self.candidates = candidates

    def index_list(self, vocab_index):
        self.i_context = [vocab_index[token] for token in self.context]
        self.i_query = [vocab_index[token] for token in self.query]
        self.i_answer = [vocab_index[token] for token in self.answer]
        self.i_candidates = [vocab_index[token] for token in self.candidates]
        return (self.i_context, self.i_query, self.i_answer, self.i_candidates)


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
    def get_cqac_words(self, cqac):
        cqac_split = cqac.split("\n21 ")
        context = self.clean(cqac_split[0])
        last_line = cqac_split[1]
        query, answer, _, candidates = [self.clean(x) for x in last_line.split("\t")]
        candidates = candidates[0].split("|")
        return context, query, answer, candidates

    def make_vocab(self, example_set):
        counter = Counter()
        print("[*] Generating vocabulary")
        for s, f_name in example_set.items():
            full_path = os.path.join(self.inner_data_dir, f_name)
            with open(full_path, 'r') as f:
                file_string = f.read()
                for cqac in file_string.split("\n\n"):
                    if len(cqac) < 5:
                        break
                    context, query, answer, _ = self.get_cqac_words(cqac)
                    for token in context + query + answer:
                        counter[token] += 1
        # Get all words in counter, and create word-id mapping
        words, _ = zip(*counter.most_common())
        vocab = {token: i for i, token in enumerate(words)}
        return vocab

    def named_entities(self, sample_dict):
        print("[*] Creating records from Named Entities")
        self.vocab = {}
        vocab_path = os.path.join("cache", "named_entities_vocab.pickle")
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as vf:
                self.vocab = pickle.load(vf)
        else:
            self.vocab = self.make_vocab(self._NAMED_ENTITY)
            with open(vocab_path, 'w') as vf:
                pickle.dump(self.vocab, vf)

        filtered_dict = {
            "train": [],
            "valid": [],
            "test": []
        }

        # Filter out existing samples
        for group, sample_list in sample_dict.items():
            for s in sample_list:
                if not os.path.exists("tfrecords/" + s.name + ".tfrecords"):
                    filtered_dict[group] += [s]

        print("[*] Applying samples")
        for s, f_name in self._NAMED_ENTITY.items():
            full_path = os.path.join(self.inner_data_dir, f_name)
            with open(full_path, 'r') as f:
                file_string = f.read()
                for i, cqac in enumerate(file_string.split("\n\n")):
                    if len(cqac) < 5:
                        break
                    context, query, answer, candidates = self.get_cqac_words(cqac)
                    print(query)
                    for sampler in filtered_dict[s]:
                        sampler(self.vocab, CBTExample(
                            i,
                            context,
                            query,
                            answer,
                            candidates
                        ))
