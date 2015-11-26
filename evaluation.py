"""
Evaluation code for multimodal ranking
Throughout, we assume 5 captions per image, and that
captions[5i:5i+5] are GT descriptions of images[i]
"""
import numpy


import datasets
from datasource import Datasource
import tools

def ranking_eval_5fold(model, split='dev'):
    """
    Evaluate a trained model on either dev or test of the dataset it was trained on
    Evaluate separately on 5 1000-image splits, and average the metrics
    """
    data = model['options']['data']
    cnn = model['options']['cnn']

    results = []

    for fold in range(5):
        print 'Loading fold ' + str(fold)
        dataset = datasets.load_dataset(data, cnn, load_train=False, fold=fold)
        caps, ims = Datasource(dataset[split], model['worddict']).all()

        print 'Computing results...'
        c_emb = tools.encode_sentences(model, caps)
        i_emb = tools.encode_images(model, ims)

        errs = tools.compute_errors(model, c_emb, i_emb)


        r = t2i(errs)
        print "Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % tuple(r)

        ri = i2t(errs)
        print "Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % tuple(ri)
        results.append(r + ri)

    print("-----------------------------------")
    print("Mean metrics: ")
    mean_metrics = numpy.array(results).mean(axis=0).flatten()
    print "Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % tuple(mean_metrics[:5])
    print "Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % tuple(mean_metrics[5:])


def t2i(c2i, vis_details=False):
    """
    Text->Images (Image Search)
    c2i: (5N, N) matrix of caption to image errors
    vis_details: if true, return a dictionary for ROC visualization purposes
    """

    ranks = numpy.zeros(c2i.shape[0])


    vis_dict = {'sentences': []}

    for i in range(len(ranks)):
        d_i = c2i[i]
        inds = numpy.argsort(d_i)

        rank = numpy.where(inds == i/5)[0][0]
        ranks[i] = rank

        def image_dict(k):
            return {'id': k, 'score': float(d_i[k])}

        if vis_details:  # save top 10 images as well as GT image and their scores
            vis_dict['sentences'].append({
                'id': i,
                'rank': rank + 1,
                'gt_image': image_dict(i/5),
                'top_images': map(image_dict, inds[:10])
            })

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1

    stats = map(float, [r1, r5, r10, medr, meanr])

    if not vis_details:
        return stats
    else:
        vis_dict['stats'] = {'R@1': r1, 'R@5': r5, 'R@10': r10, 'median_rank': medr, 'mean_rank': meanr}
        return stats, vis_dict


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
    return map(float, [r1, r5, r10, medr, meanr])
