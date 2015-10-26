import numpy
import copy
import sys
from collections import defaultdict

class HierarchyData():

    def __init__(self, data, worddict, batch_size=128, maxlen=None, n_words=10000):
        self.data = data
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.worddict = worddict
        self.n_words = n_words

        self.ancestors = defaultdict(set)
        self.num_vertices = len(data['caps'])
        if 'ims' in data:
            self.num_vertices += len(data['ims'])

        self.prepare()
        self.reset()

    def prepare(self):

        # compute adjacency list
        edges = self.data['edges']

        vertices = set()
        parents = defaultdict(set)
        for edge in edges:
            parents[edge[0]].add(edge[1])
            vertices.add(edge[0])
            vertices.add(edge[1])




        # compute transitive closure of parent relation

        def dfs(start, cur):
            if start != cur:
                self.ancestors[start].add(cur)

            for next in parents[cur]:
                dfs(start, next)

        for vertex in vertices:
            dfs(vertex, vertex)


    def contrastive_negatives(self, edges, max_index):
        """ generate negatives by randomly replacing one of the indices in each edge"""
        N = edges.shape[0]
        r = numpy.random.rand(N) > 0.5
        mask = numpy.vstack((r, r == 0)).T
        random_indices = numpy.random.randint(0, max_index, size=N)
        negs = numpy.copy(edges)
        negs[mask] = random_indices
        return negs



    def random_negatives(self, edges, max_index):
        """ returns a list of random negative examples """
        return numpy.random.randint(0, max_index, size=edges.shape)






    def up_closure(self, indices):
        """ returns up-closure of the given list of indices, under the hierarchy, as well as all edges,
        with edge indices local to the returned up-closure """
        closure = set(indices)
        edges = set()
        for index in indices:
            closure |= self.ancestors[index]
            for a in self.ancestors[index]:
                edges.add((index, a))

        closure = list(closure)
        to_local = dict(map(reversed, enumerate(closure)))

        edges = [(to_local[i], to_local[j]) for (i, j) in edges]

        return closure, edges




    def reset(self):
        self.idx = 0
        self.order = numpy.random.permutation(self.num_vertices)



    def next(self):
        indices = []

        while len(indices) < self.batch_size:
            indices.append(self.order[self.idx])
            self.idx += 1
            if self.idx >= self.num_vertices:
                self.reset()
                raise StopIteration()


        indices, edges = self.up_closure(indices)
        edges = numpy.array(edges)
        negatives = self.contrastive_negatives(edges, len(indices))

        x, x_mask = self.prepare_caps(indices)

        return x, x_mask, None, edges, negatives


    def all(self):
        indices, edges = self.up_closure(list(range(self.num_vertices)))
        edges = numpy.array(edges)
        negatives = self.contrastive_negatives(edges, len(indices))

        caps = []
        for i in indices:
            caps.append(self.data['caps'][i])

        return caps, edges, negatives


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

        # 1 means use, 0 means skip (also end-of-sentence token)
        x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
        for idx, s in enumerate(seqs):
            x[:lengths[idx],idx] = s
            x_mask[:lengths[idx]+1, idx] = 1.

        return x, x_mask

