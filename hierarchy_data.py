import numpy
import copy
import sys
from collections import defaultdict, deque
import random

class HierarchyData():

    def __init__(self, data, worddict, batch_size=128, maxlen=None, n_words=10000, max_edges_per_batch=None, max_nodes_per_batch=None):
        self.data = data
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.max_nodes_per_batch = max_nodes_per_batch
        self.worddict = worddict
        self.n_words = n_words
        self.max_edges_per_batch = max_edges_per_batch
        self.num_captions = len(self.data['caps'])
        self.image_ids = range(self.num_captions, self.num_captions + len(self.data['ims']))

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
        N = len(edges)
        r = numpy.random.rand(N) > 0.5
        mask = numpy.vstack((r, r == 0)).T
        random_indices = numpy.random.randint(0, max_index, size=N)
        negs = numpy.copy(edges)
        negs[mask] = random_indices

        edge_set = set((edges[i][0], edges[i][1]) for i in range(N))
        indices = [i for i in range(N) if (negs[i][0], negs[i][1]) not in edge_set]

        return negs[indices]



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

        # compute all captions above the given image ids
        closure = set()

        # recursive algorithm, with caching
        ancestors = dict()

        def getAncestors(i, choice=False):
            if i not in ancestors:
                if choice:
                    ancestors[i] = {random.choice(list(self.parents[i]))}
                else:
                    ancestors[i] = set(self.parents[i])
                #for n in self.parents[i]:
                #    ancestors[i] |= getAncestors(n)

            return ancestors[i]

        for index in image_ids:
            closure |= getAncestors(index, only_one_caption)

        closure = list(closure)
        if max_indices is not None:
            del closure[max_indices:]

        closure += image_ids  # put image ids at the end

        to_local = dict(map(reversed, enumerate(closure)))

        # get edges
        edges = [(to_local[i], to_local[a]) for (i, As) in ancestors.iteritems() for a in As if i in to_local and a in to_local]

        return closure, edges, to_local


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
        all_ids, edges, _ = self.up_closure(image_ids, self.max_nodes_per_batch)
        edges = numpy.array(edges)
        negatives = self.contrastive_negatives(edges, len(all_ids))

        if self.max_edges_per_batch is not None:
            edges = edges[:self.max_edges_per_batch]
            negatives = negatives[:self.max_edges_per_batch]

        caption_ids, image_ids = self.split_modes(all_ids)
        x, x_mask = self.prepare_caps(caption_ids)
        im = self.data['ims'][numpy.array(image_ids)]

        return x, x_mask, im, edges, negatives


    def all(self):
        all_ids, edges, to_local = self.up_closure(self.image_ids)
        edges = numpy.array(edges)
        negatives = self.contrastive_negatives(edges, len(all_ids))

        caption_ids, image_ids = self.split_modes(all_ids)
        caps = []
        for i in caption_ids:
            caps.append(self.data['caps'][i])

        target = numpy.hstack((numpy.ones(edges.shape[:1]), numpy.zeros(negatives.shape[:1])))
        edges = numpy.vstack((edges, negatives))

        # compute the 5N*N edges between all captions and all images
        c_ids = [to_local[c] for cs in self.data['image2caption'] for c in cs]
        i_ids = [to_local[i] for i in self.image_ids]
        rank_edges = numpy.array([[i, c] for c in c_ids for i in i_ids])

        return caps, self.data['ims'], edges, target, rank_edges


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

