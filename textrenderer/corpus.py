from abc import abstractmethod
import numpy as np
import random
import glob

from libs.utils import prob, load_chars


class Corpus(object):
    def __init__(self, chars_file, corpus_dir=None, length=None):
        self.corpus_dir = corpus_dir
        self.length = length

        self.chars_file = chars_file
        self.charsets = load_chars(chars_file)

        # wz
        self.frequency = {}
        for char in self.charsets: self.frequency.update({char: 0})

        if not isinstance(self, RandomCorpus):
            print("Loading corpus from: " + self.corpus_dir)

        self.load()

    @abstractmethod
    def load(self):
        """
        Read corpus from disk to memory
        """
        pass

    @abstractmethod
    def get_sample(self):
        """
        Get word line from corpus in memory
        :return: string
        """
        pass

    # wz
    def get_frequency(self):
        return self.frequency

    def get_charset(self):
        return self.frequency


class RandomCorpus(Corpus):
    """
    Load charsets and generate random word line from charsets
    """
    def load(self):
        pass

    def get_sample(self):
        word = ''
        for _ in range(self.length):
            char = random.choice(self.charsets)
            word += char
            self.frequency[char] += 1
        return word



class EngCorpus(Corpus):
    def load(self):
        corpus_path = glob.glob(self.corpus_dir + '/*.txt')
        self.corpus = []
        for i in range(len(corpus_path)):
            print("Load {}th eng corpus".format(i))
            with open(corpus_path[i], encoding='utf-8') as f:
                data = f.read()

            for word in data.split(' '):
                word = word.strip()
                word = ''.join(filter(lambda x: x in self.charsets, word))

                if word != u'' and len(word) > 2:
                    self.corpus.append(word)
            print("Word count {}".format(len(self.corpus)))

    def get_sample(self):
        word1 = random.choice(self.corpus)
        word2 = random.choice(self.corpus)

        word = ' '.join([word1, word2])
        for char in word: self.frequency[char] += 1 #wz
        return word


class ChnCorpus(Corpus):
    def load(self):
        """
        Load one corpus file as one line
        """
        corpus_path = glob.glob(self.corpus_dir + '/*.txt')
        self.corpus = []
        for i in range(len(corpus_path)):
            print_end = '\n' if i == len(corpus_path) - 1 else '\r'
            print("Loading chn corpus: {}/{}".format(i + 1, len(corpus_path)), end=print_end)
            with open(corpus_path[i], encoding='utf-8') as f:
                data = f.readlines()

            lines = []
            for line in data:
                line_striped = line.strip()
                line_striped = line_striped.replace('\u3000', '')
                line_striped = line_striped.replace('&nbsp', '')
                line_striped = line_striped.replace("\00", "")

                if line_striped != u'' and len(line.strip()) > 1:
                    lines.append(line_striped)

            # 所有行合并成一行
            split_chars = [',', '，', '：', '-', ' ', ';', '。']
            splitchar = random.choice(split_chars)
            whole_line = splitchar.join(lines)

            # 在 crnn/libs/label_converter 中 encode 时还会进行过滤
            whole_line = ''.join(filter(lambda x: x in self.charsets, whole_line))

            if len(whole_line) > self.length:
                self.corpus.append(whole_line)

    def get_sample(self):
        # 每次 gen_word，随机选一个预料文件，随机获得长度为 word_length 的字符
        line = random.choice(self.corpus)

        start = np.random.randint(0, len(line) - self.length)

        word = line[start:start + self.length]
        for char in word: self.frequency[char] += 1 #wz
        return word
