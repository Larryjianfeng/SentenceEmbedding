import os
import itertools
import numpy as np
import tensorflow as tf
import pickle
from collections import Counter



class TextReader(object):
  def __init__(self, data_path):
    train_path = os.path.join(data_path, "train.txt")
    vocab_path = os.path.join(data_path, "vocab")

    if os.path.exists(vocab_path):
      self.vocab = pickle.load(open(vocab_path, 'r'))
      self.train_data = self._file_to_data(train_path)

    else:
      self._build_vocab(train_path, vocab_path)
      self.train_data = self._file_to_data(train_path)

    self.idx2word = {v:k for k, v in self.vocab.items()}
    self.vocab_size = len(self.vocab)
    self.i = 0



  def _build_vocab(self, file_path, vocab_path):
    d = open(file_path).read().split('\t')
    counter = Counter([i for line in d for i in line.split(' ')])

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    count_pairs = [i for i in count_pairs if i[1] >=1]
    words, _ = list(zip(*count_pairs))
    self.vocab = dict(zip(words, range(len(words))))
    print 'the coabsize is\t' + str(len(self.vocab))

    pickle.dump(self.vocab, open(vocab_path, 'w'))

  def _file_to_data(self, file_path):
    texts = open(file_path).read().split('\n')
    np.random.shuffle(texts)
    data = []
    self.texts = []
    for text in texts:
      trans = [i for i in np.array(map(self.vocab.get, text.split())) if i!=None]
      if len(trans) >= 4:
        data.append(list(set(trans)))
        self.texts.append(text)
    return data

  def onehot(self, data):
    return np.bincount(data, minlength = self.vocab_size)


  def iterator(self, data_type="train"):

    if self.i + 1000 >= len(self.train_data):
      self.i  = len(self.train_data)%1000
      x = self.train_data[self.i: self.i+1000]
    else:
      x = self.train_data[self.i: self.i+1000]
      self.i += 1000

    x = [self.onehot(i) for i in x]
    return x, len(x)


  def get(self, text):
    text = text.split(' ')
    data = np.array(map(self.vocab.get, text))
    data = [i for i in data if i!=None]
    return [self.onehot(data)]

  def random(self, data_type="train"):
    raw_data = self.train_data
    idx = np.random.randint(len(raw_data))

    data = raw_data[idx]
    return self.onehot(data)
