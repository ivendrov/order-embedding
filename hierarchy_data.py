import numpy
import copy
import sys
from collections import defaultdict, deque

class HierarchyData():

    def __init__(self, data, worddict, batch_size=128, maxlen=None, n_words=10000, max_edges_per_batch=None, max_nodes_per_batch=None):
        self.data = data
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.max_nodes_per_batch = max_nodes_per_batch
        self.worddict = worddict
        self.n_words = n_words
        self.max_edges_per_batch = max_edges_per_batch
        self.leaves = []

        self.parents = defaultdict(set)
        self.num_vertices = len(data['caps'])
        if 'ims' in data:
            self.num_vertices += len(data['ims'])
        self.prepare()
        self.reset()


    def change_maxlen(self, maxlen):
        self.maxlen = maxlen
        self.prepare()
        self.reset()

    def prepare(self):
        # compute set of vertices with len <= max_len
        allowed = set()

        if self.maxlen is None:
            allowed.update(range(self.num_vertices))
        else:
            allowed.update(i for (i, cap) in enumerate(self.data['caps']) if len(cap.split()) <= self.maxlen)



        print("Preparing data...")
        # compute adjacency list, and leaves
        edges = self.data['edges']
        leaves = set(edge[0] for edge in edges if edge[0] in allowed)

        for edge in edges:
            if edge[0] in allowed and edge[1] in allowed:
                self.parents[edge[0]].add(edge[1])
                if edge[1] in leaves:
                    leaves.remove(edge[1])

        self.leaves = list(leaves)
        print("Split has " + str(len(self.leaves)) + " leaves")
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






    def up_closure(self, indices, max_indices=None):
        """ returns up-closure of the given list of indices, under the hierarchy, as well as all edges,
        with edge indices local to the returned up-closure """

        closure = set(indices)  # maintain dict of global index to local index

        # recursive algorithm, with caching
        ancestors = dict()


        def getAncestors(i):
            if i not in ancestors:
                ancestors[i] = set(self.parents[i])
                for n in self.parents[i]:
                    ancestors[i] |= getAncestors(n)

            return ancestors[i]

        for index in indices:
            closure |= getAncestors(index)

        closure = list(closure)
        if max_indices is not None:
            del closure[max_indices:]

        to_local = dict(map(reversed, enumerate(closure)))

         # get edges
        edges = [(to_local[i], to_local[a]) for (i, As) in ancestors.iteritems() for a in As if i in to_local and a in to_local]

        return closure, edges


    def reset(self):
        self.idx = 0
        self.order = numpy.random.permutation(len(self.leaves))



    def next(self):
        indices = []

        while len(indices) < self.batch_size:
            indices.append(self.leaves[self.order[self.idx]])
            self.idx += 1
            if self.idx >= len(self.leaves):
                self.reset()
                raise StopIteration()


        indices, edges = self.up_closure(indices, self.max_nodes_per_batch)
        edges = numpy.array(edges)
        negatives = self.contrastive_negatives(edges, len(indices))

        if self.max_edges_per_batch is not None:
            edges = edges[:self.max_edges_per_batch]
            negatives = negatives[:self.max_edges_per_batch]

        x, x_mask = self.prepare_caps(indices)
        return x, x_mask, None, edges, negatives


    def all(self):
        indices, edges = self.up_closure(self.leaves)
        edges = numpy.array(edges)
        negatives = self.contrastive_negatives(edges, len(indices))

        caps = []
        for i in indices:
            caps.append(self.data['caps'][i])

        target = numpy.hstack((numpy.ones(edges.shape[:1]), numpy.zeros(negatives.shape[:1])))
        edges = numpy.vstack((edges, negatives))

        return caps, edges, target


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

