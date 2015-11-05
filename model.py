"""
Model specification
"""
import theano
import theano.tensor as tensor
from theano.tensor.extra_ops import fill_diagonal
import numpy

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.ifelse import ifelse

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
    params = get_layer('ff')[0](options, params, prefix='ff_image', nin=options['dim_image'], nout=options['dim'])

    return params

def hierarchical_errors(s, im, options):
    return tensor.pow(tensor.maximum(0, s - im + options['eps']), options['norm'])

def symmetric_loss(s, im, options):
    im = l2norm(im)

    if options['abs']:
        im = abs(im)
        s = abs(s)
    margin = options['margin']

    im = im.dimshuffle(('x', 0, 1))
    s = s.dimshuffle((0, 'x', 1))
    scores = hierarchical_errors(s, im, options).sum(axis=2)

    diagonal = scores.diagonal()

    # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
    cost_s = tensor.maximum(0, margin - scores + diagonal)
    cost_im = tensor.maximum(0, margin - scores + diagonal.reshape((-1,1)))

    # clear diagonals
    cost_tot = cost_s + cost_im
    cost_tot = fill_diagonal(cost_tot, 0)


    return cost_tot.sum() + 2 * diagonal.sum()



def build_model(tparams, options):
    """
    Computation graph for the model
    """
    opt_ret = dict()
    trng = RandomStreams(1234)

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')
    im = tensor.matrix('im', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # Word embedding (source)
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

    # Encode sentences (source)
    proj = get_layer(options['encoder'])[1](tparams, emb, None, options,
                                            prefix='encoder',
                                            mask=mask)
    sents = proj[0][-1]

    # Encode images (source)
    images = get_layer('ff')[1](tparams, im, options, prefix='ff_image', activ='linear')

    # Compute loss
    cost = symmetric_loss(sents, images, options)

    return trng, [x, mask, im], cost



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
    sents = proj[0][-1]

    if options['abs']:
        sents = abs(sents)

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
    images = l2norm(images)

    if options['abs']:
        images = abs(images)
    
    return trng, [im], images


def build_errors(options):
    """ Given sentence and image embeddings, compute the score matrix """
    # input features
    s = tensor.matrix('s', dtype='float32')
    im = tensor.matrix('im', dtype='float32')

    # trick to make theano not optimize this into a single matrix op, and overflow memory
    indices = tensor.arange(s.shape[0])

    # have to do a map in order not to overflow memory here
    errs, _ = theano.map(lambda i, s, im: hierarchical_errors(s[i], im, options).sum(axis=1).flatten(),
                      sequences=[indices],
                      non_sequences=[s, im])

    return [s, im], errs



