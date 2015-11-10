"""
Evaluation code for multimodal-ranking
"""
import numpy


import datasets
from hierarchy_data import HierarchyData
import tools

def evalrank(model, split='dev'):
    """
    Evaluate a trained model on either dev or test of the dataset it was trained on
    """
    dataset = model['options']['data']
    cnn = model['options']['cnn']
    captions = model['options']['captions']

    print 'Loading dataset...'
    dataset = datasets.load_dataset(dataset, cnn, captions, load_train=False)
    caps, ims = HierarchyData(dataset[split], model['worddict'], n_words=len(model['word_dict']))  # TODO what is n_words??

    print 'Computing results...'
    c_emb = tools.encode_sentences(model, caps)
    i_emb = tools.encode_images(model, ims)
    errs = tools.compute_errors(model, c_emb, i_emb)


    (r1, r5, r10, medr, meanr) = t2i(errs)
    print "Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr)

    (r1i, r5i, r10i, medri, meanri) = i2t(errs)
    print "Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanri)

    return r1i, r5i, r10i, medri, meanri, r1, r5, r10, medr, meanr

def t2i(c2i):
    """
    Text->Images (Image Search)
    c2i: (5N, N) matrix of caption to image errors
    """

    ranks = numpy.zeros(c2i.shape[0])

    for i in range(len(ranks)):
        d_i = c2i[i]
        inds = numpy.argsort(d_i)

        rank = numpy.where(inds == i/5)[0][0]
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return (r1, r5, r10, medr, meanr)


def i2t(c2i):
    """
    Text->Images (Image Search)
    c2i: (5N, N) matrix of caption to image errors
    """

    ranks = numpy.zeros(c2i.shape[1])

    for i in range(len(ranks)):
        d_i = c2i[:, i]
        inds = numpy.argsort(d_i)

        rank = numpy.where(inds/5 == i)[0][0]
        ranks[i] = rank


    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return (r1, r5, r10, medr, meanr)
