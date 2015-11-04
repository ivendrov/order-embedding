import numpy
import copy
import sys
from collections import defaultdict, deque
import random

class HierarchyData():

    def __init__(self, data, worddict, batch_size=128, maxlen=None, n_words=10000, max_edges_per_batch=None, max_nodes_per_batch=None, num_contrastive=1, onlycaps=False):
        self.data = data
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.num_contrastive = num_contrastive
        self.max_nodes_per_batch = max_nodes_per_batch
        self.worddict = worddict
        self.n_words = n_words
        self.max_edges_per_batch = max_edges_per_batch
        self.num_captions = len(self.data['caps'])
        self.image_ids = range(self.num_captions, self.num_captions + len(self.data['ims']))
        self.onlycaps = onlycaps
        self.parents = defaultdict(set)
        self.prepare()
        self.reset()

    def isImage(self, index):
        return index >= self.num_captions
    def isCaption(self, index):
        return not self.isImage(index)

    def split_modes(self, indices):
        """ split the given combined indices into caption and image indices """
        caption_ids = filter(self.isCaption, indices)
        image_ids = map(lambda i: i - self.num_captions, filter(self.isImage, indices))
        return caption_ids, image_ids


    def change_maxlen(self, maxlen):
        self.maxlen = maxlen
        self.prepare()
        self.reset()

    def prepare(self):
        # 1. compute parents relation:
        #   a. edges between text nodes
        if not self.onlycaps:
            for edge in self.data['edges']:
                self.parents[edge[0]].add(edge[1])

        #   b. edges between images and captions
        for i, caps in enumerate(self.data['image2caption']):
            for cap in caps:
                self.parents[self.image_ids[i]].add(cap)

        print("Split has " + str(len(self.data['ims'])) + " images")
        print("Done")



    def contrastive_negatives(self, edges, max_index):
        """ generate negatives by randomly replacing one of the indices in each edge"""
        negs_list = []
        for k in range(self.num_contrastive):
            N = len(edges)
            r = numpy.random.rand(N) > 0.5
            mask = numpy.vstack((r, r == 0)).T
            random_indices = numpy.random.randint(0, max_index, size=N)
            negs = numpy.copy(edges)
            negs[mask] = random_indices

            edge_set = set((edges[i][0], edges[i][1]) for i in range(N))
            indices = [i for i in range(N) if (negs[i][0], negs[i][1]) not in edge_set]
            negs_list.append(negs[indices])

        return numpy.vstack(negs_list)



    def random_negatives(self, edges, max_index):
        """ returns a list of random negative examples """
        return numpy.random.randint(0, max_index, size=edges.shape)






    def up_closure(self, image_ids, max_indices=None, only_one_caption=False):
        """ returns the first max_indices (or all) caption ids above the given image ids, under the hierarchy, as well as all edges,
            with edge indices local to the returned array:
                0 .. len(caption_ids) for captions,
                len(caption_ids) .. len(caption_ids) .. len(image_ids) for images

            Also returns the global-to-local mapping
        """
        caps = map(lambda i: random.choice(list(self.parents[i])), image_ids)

        return caps


    def reset(self):
        self.idx = 0
        self.order = numpy.random.permutation(len(self.image_ids))



    def next(self):
        indices = []

        while len(indices) < self.batch_size:
            indices.append(self.image_ids[self.order[self.idx]])
            self.idx += 1
            if self.idx >= len(self.image_ids):
                self.reset()
                raise StopIteration()

        image_ids = indices
        caption_ids = self.up_closure(image_ids)
        image_ids = map(lambda i: i - self.num_captions, image_ids)

        x, x_mask = self.prepare_caps(caption_ids)
        im = self.data['ims'][numpy.array(image_ids)]

        return x, x_mask, im


    def all(self):
        caps = []
        for i, cs in enumerate(self.data['image2caption']):
            for cap in cs:
                caps.append(self.data['caps'][cap])

        print("Number of captions in dataset: " + str(len(caps)))


        return caps, self.data['ims']

    def __iter__(self):
        return self


    def prepare_caps(self, indices):
        seqs = []
        for i in indices:
            cc = self.data['caps'][i]
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

