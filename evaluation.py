"""
Evaluation code for multimodal-ranking
"""
import numpy

from datasets import load_dataset
from tools import encode_sentences, encode_images

def evalrank(model, data, split='dev'):
    """
    Evaluate a trained model on either dev or test
    data options: f8k, f30k, coco
    """
    print 'Loading dataset'
    dataset = load_dataset(data, load_train=False)[split]

    print 'Computing results...'
    ls = encode_sentences(model, dataset['caps'])
    lim = encode_images(model, dataset['ims'])

    #(r1, r5, r10, medr) = i2t(lim, ls)
    #print "Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr)
    (r1i, r5i, r10i, medri) = t2i(lim, ls)
    print "Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri)


def hierachical_error(e):
    return numpy.linalg.norm(numpy.maximum(e[:, 1, :] - e[:, 0, :], 0), ord=1, axis=1)

def errors(s, pos, neg):
    target = numpy.hstack((numpy.ones(pos.shape[:1]), numpy.zeros(neg.shape[:1])))
    edges = numpy.vstack((pos, neg))
    errs = hierachical_error(s[edges])
    return target, errs



def best_threshold(s, pos, neg):
    target, errs = errors(s, pos, neg)
    indices = numpy.argsort(errs)
    sortedErrors = errs[indices]
    sortedTarget = target[indices]
    tp = numpy.cumsum(sortedTarget)
    invSortedTarget = (sortedTarget == 0).astype('float32')
    Nneg = invSortedTarget.sum()
    fp = numpy.cumsum(invSortedTarget)
    tn = fp * -1 + Nneg
    accuracies = (tp + tn) / sortedTarget.shape[0]
    i = accuracies.argmax()
    print("Number of positives, negatives, tp, tn: %f %f %f %f" % (target.sum(), Nneg, tp[i], tn[i]))
    return sortedErrors[i]



def eval_accuracy(s1, p1, n1, s2, p2, n2):
    thresh = best_threshold(s1, p1, n1)

    target, errs = errors(s2, p2, n2)
    pred = errs <= thresh

    accuracy = float((pred == target).astype('float32').mean())

    return accuracy


def i2t(images, captions, npts=None):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts == None:
        npts = images.shape[0] / 5
    index_list = []

    ranks = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        d = numpy.linalg.norm(numpy.maximum(0, captions - im), ord=1, axis=1).flatten()
        inds = numpy.argsort(d)
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5*index, 5*index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)

def t2i(images, captions, npts=None):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts == None:
        npts = images.shape[0] / 5
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])
    ims = numpy.expand_dims(ims, 0)

    num_zero = 0

    ranks = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5*index : 5*index + 5]
        queries = numpy.expand_dims(queries, 1)

        # Compute scores
        d = numpy.linalg.norm(numpy.maximum(0, queries - ims), ord=1, axis=2)

        for i in range(len(d)):
            d_i = d[i]
            inds = numpy.argsort(d_i)

            ranks[5 * index + i] = numpy.where(inds == index)[0][0]
            rank = ranks[5*index + i]

            if d_i[inds[rank]] == 0:
                num_zero += 1


    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    print("Fraction of GT pairs with score zero: " + str(num_zero) + " / " + str(captions.shape[0]))
    return (r1, r5, r10, medr)
