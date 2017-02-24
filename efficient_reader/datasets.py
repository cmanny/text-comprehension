import os
import requests
import tarfile


class CBTDataSet(object):
    def __init__(self, data_dir, in_memory=False, name="dts", *args, **kwargs):
        self.in_memory = in_memory
        self.data_dir = data_dir
        self.name = name

    # Function to automatically handle data
    def auto_setup(self):
        self.from_url("http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz")

    def from_url(self, url):
        self.inner_data = os.path.join(self.data_dir, self.name)
        print(self.inner_data)
        if os.exists(self.inner_data):
            return
        file_name = os.path.join(self.data_dir, self.name + ".tar.gz")
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
            directory = os.path.join(self.data_dir, self.name)
            os.mkdir(directory)
            tf.extractall(directory)

    # I really shoudn't do this
    def get_ne_data():
        data_dir = os.join(self.inner_data, "CBTest", "data")
        with open(os.join(data_dir, "cbtest_NE_test_2500ex.txt")) as test:
            self.test = test.readlines()

        with open(os.join(data_dir, "cbtest_NE_train.txt")) as train:
            self.train = train.readlines()

        with open(os.join(data_dir, "cbtest_NE_valid_2000ex.txt")) as valid:
            self.valid = valid.readlines()
