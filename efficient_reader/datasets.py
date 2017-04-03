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

    def __init__(self, data_dir="raw_data", in_memory=False, name="cbt_data",
                 *args, **kwargs):
        self.in_memory = in_memory
        self.top_data_dir = data_dir
        self.name = name
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

    # def _vocab(self, vocab_file):
    #     if os.path.exists(vocab_file):
    #         with open(vocab_file, 'r') as f:
    #             return pickle.load(f)
    #
    #     directories = ['data/train/', 'data/valid/', 'data/test/']
    #     files = [directory + file_name for directory in directories \
    #              for file_name in os.listdir(directory)]
    #     counter = Counter()
    #     for file_name in files:
    #         with open(file_name, 'r') as f:
    #             file_string = f.read()
    #             for cqa in file_string.split("\n\n"):
    #                     if len(cqa) < 5:
    #                         break
    #                     context, query, answer = get_cqa_words(cqa)
    #                     for token in context + query + answer:
    #                         counter[token] += 1
    #     with open(vocab_file, 'w') as f:
    #         pickle.dump(counter, f)
    #     return counter

    @classmethod
    def clean(self, string):
        return [re.sub("[\n0-9]", '', x) for x in string.split(" ")]

    def get_cqa_words(self, cqa):
        cqa_split = cqa.split("\n21 ")
        context = clean(cqa_split[0])
        last_line = cqa_split[1]
        query, answer = [clean(x) for x in last_line.split("\t", 2)[:2]]
        return context, query, answer

    def named_entities(self, record_folder, perc=1.0, sort_asc=1):
        directories = [os.path.join(record_folder, f) \
                       for f in self._NAMED_ENTITY.values()]
        if os.path.exists("ne_vocab"):
            with open("ne_vocab", 'r') as f:
                self.ne_vocab = pickle.load(f)
        else:
            self.ne_vocab = None
        self.tokenize(self.index, self.word)




    def tokenize(self, index, word):
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

    def run(self):
        counter = counts()
        print('num words',len(counter))
        word, _ = zip(*counter.most_common())
        index = {token: i for i, token in enumerate(word)}
        tokenize(index, word)
        print('DONE')
