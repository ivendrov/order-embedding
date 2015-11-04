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


def best_threshold(errs, target):
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



def eval_accuracy(e1, t1, e2, t2):
    # find best threshold on the first dev set, use it to evaluate accuracy on the second
    thresh = best_threshold(e1, t1)
    pred = e2 <= thresh
    correct = (pred == t2)

    accuracy = float(correct.astype('float32').mean())

    wrong_indices = numpy.logical_not(correct).nonzero()[0]
    wrong_preds = pred[wrong_indices]

    return accuracy, wrong_indices, wrong_preds


def t2i(c2i):
    """
    Text->Images (Image Search)
    c2i: (5N, N) matrix of caption to image errors
    """

    num_zero = 0

    ranks = numpy.zeros(c2i.shape[0])

    for i in range(len(ranks)):
        d_i = c2i[i]
        inds = numpy.argsort(d_i)

        rank = numpy.where(inds == i/5)[0][0]
        ranks[i] = rank

        if d_i[inds[rank]] == 0:
            num_zero += 1


    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    print("Fraction of GT pairs with score zero: " + str(num_zero) + " / " + str(c2i.shape[0]))
    return (r1, r5, r10, medr, meanr)
