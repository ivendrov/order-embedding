import numpy
from collections import defaultdict
import random

class Datasource():
    """
    Wrapper around a dataset which permits

    1) Iteration over minibatches using next(); call reset() between epochs to randomly shuffle the data
    2) Access to the entire dataset using all()
    """

    def __init__(self, data, worddict, batch_size=128):
        self.data = data
        self.batch_size = batch_size
        self.worddict = worddict
        self.num_images = len(self.data['ims'])
        self.parents = defaultdict(set)
        self.reset()

    def reset(self):
        self.idx = 0
        self.order = numpy.random.permutation(self.num_images)

    def next(self):
        image_ids = []
        caption_ids = []

        while len(image_ids) < self.batch_size:
            image_id = self.order[self.idx]
            caption_id = image_id * 5 + random.randrange(5)
            image_ids.append(image_id)
            caption_ids.append(caption_id)

            self.idx += 1
            if self.idx >= self.num_images:
                self.reset()
                raise StopIteration()

        x, x_mask = self.prepare_caps(caption_ids)
        im = self.data['ims'][numpy.array(image_ids)]

        return x, x_mask, im

    def all(self):
        return self.data['caps'], self.data['ims']

    def __iter__(self):
        return self

    def prepare_caps(self, indices):
        seqs = []
        for i in indices:
            cc = self.data['caps'][i]
            seqs.append([self.worddict[w] if w in self.worddict else 1 for w in cc.split()])

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

