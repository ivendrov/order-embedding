import numpy
import copy
import sys
from collections import defaultdict, deque

class HierarchyData():

    def __init__(self, data, worddict, batch_size=128, maxlen=None, n_words=10000):
        self.data = data
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.worddict = worddict
        self.n_words = n_words
        self.leaves = []

        self.parents = defaultdict(set)
        self.num_pairs = len(data['caps'])
        print("Loaded dataset with " +  str(self.num_pairs) + " pairs of captions")
        self.reset()







    def reset(self):
        self.idx = 0
        self.order = numpy.random.permutation(self.num_pairs)


    def caps_for_indices(self, indices):
        return [self.data['caps'][i][j] for j in range(2) for i in indices]



    def next(self):
        indices = []

        while len(indices) < self.batch_size:
            indices.append(self.order[self.idx])
            self.idx += 1
            if self.idx >= self.num_pairs:
                self.reset()
                raise StopIteration()

        x, x_mask = self.prepare_caps(self.caps_for_indices(indices))
        return x, x_mask, self.data['labels'][indices]


    def all(self):
        indices = range(self.num_pairs)

        caps = self.caps_for_indices(indices)

        return caps, self.data['labels']


    def __iter__(self):
        return self


    def prepare_caps(self, caps):
        seqs = []
        for cc in caps:
            seqs.append([self.worddict[w] if self.worddict[w] < self.n_words else 1 for w in cc.split()])

        lengths = map(len, seqs)
        n_samples = len(seqs)
        maxlen = numpy.max(lengths) + 1

        x = numpy.zeros((maxlen, n_samples)).astype('int64')

        # 1 means use, 0 means skip
        x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
        for idx, s in enumerate(seqs):
            x[:lengths[idx], idx] = s
            x_mask[:lengths[idx]+1, idx] = 1.

        return x, x_mask

