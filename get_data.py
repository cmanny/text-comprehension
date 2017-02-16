import os
import requests
import tarfile

class CBTDataSet(object):
    def __init__(self, data_dir, in_memory=False, name="Data", *args, **kwargs):
        self.in_memory = in_memory
        self.data_dir = data_dir
        self.name = name

    def from_url(self, url):
        file_name = os.path.join(self.data_dir, self.name + ".tar.gz")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:10.0) Gecko/20100101 Firefox/10.0',
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

if __name__ == "__main__":
    d = CBTDataSet("data", name="cbt_data")
    d.from_url("http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz")
