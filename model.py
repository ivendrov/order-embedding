"""
Model specification
"""
import theano
import theano.tensor as tensor
from theano.tensor.extra_ops import fill_diagonal
import numpy

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import _p, ortho_weight, norm_weight, xavier_weight, tanh, l2norm
from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer

def init_params(options):
    """
    Initialize all parameters
    """
    params = OrderedDict()

    # Word embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

    # Sentence encoder
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder',
                                              nin=options['dim_word'], dim=options['dim'])

    # Image encoder
    # params = get_layer('ff')[0](options, params, prefix='ff_image', nin=options['dim_image'], nout=options['dim'])

    return params

def hierarchical_error(edges):
    specific = edges[:, 0, :]
    general = edges[:, 1, :]
    return tensor.maximum(0, general-specific + 1e-5).norm(1, 1)

def contrastive_loss(margin, s, edges, negatives):
    """
    Compute contrastive loss
    """
    pos = s[edges]
    neg = s[negatives]

    pos_costs = hierarchical_error(pos)
    neg_costs = tensor.maximum(0, margin - hierarchical_error(neg))

    return (pos_costs.sum() + neg_costs.sum()) / (edges.shape[0] + negatives.shape[0])


def build_model(tparams, options):
    """
    Computation graph for the model
    """
    opt_ret = dict()
    trng = RandomStreams(1234)

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')
    edges = tensor.matrix('edges', dtype='int64')
    negatives = tensor.matrix('negatives', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # Word embedding (source)
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

    # Encode sentences (source)
    proj = get_layer(options['encoder'])[1](tparams, emb, None, options,
                                            prefix='encoder',
                                            mask=mask)
    sents = abs(proj[0][-1])

    # Compute loss
    cost = contrastive_loss(options['margin'], sents, edges, negatives)

    return trng, [x, mask, edges, negatives], cost


def build_errors(tparams, options):
    feats = tensor.matrix('feats', dtype='float32')
    edges = tensor.matrix('edges', dtype='int64')

    errors = hierarchical_error(feats[edges])
    return [feats, edges], errors


def build_sentence_encoder(tparams, options):
    """
    Encoder only, for sentences
    """
    opt_ret = dict()

    trng = RandomStreams(1234)

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('x_mask', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # Word embedding
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

    # Encode sentences
    proj = get_layer(options['encoder'])[1](tparams, emb, None, options,
                                            prefix='encoder',
                                            mask=mask)
    sents = abs(proj[0][-1])

    return trng, [x, mask], sents

def build_image_encoder(tparams, options):
    """
    Encoder only, for images
    """
    opt_ret = dict()

    trng = RandomStreams(1234)

    # image features
    im = tensor.matrix('im', dtype='float32')

    # Encode images
    images = get_layer('ff')[1](tparams, im, options, prefix='ff_image', activ='linear')
    images = abs(images)
    
    return trng, [im], images


