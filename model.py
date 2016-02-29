"""
Model specification
"""
import theano
import theano.tensor as tensor
from theano.tensor.extra_ops import fill_diagonal

from collections import OrderedDict

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


def order_violations(s, im, options):
    """
    Computes the order violations (Equation 2 in the paper)
    """
    return tensor.pow(tensor.maximum(0, s - im), 2)


def contrastive_loss(s, im, options):
    """
    For a minibatch of sentence and image embeddings, compute the pairwise contrastive loss
    """
    margin = options['margin']

    if options['method'] == 'order':
        im2 = im.dimshuffle(('x', 0, 1))
        s2 = s.dimshuffle((0, 'x', 1))
        errors = order_violations(s2, im2, options).sum(axis=2)
    elif options['method'] == 'cosine':
        errors = - tensor.dot(im, s.T) # negative because error is the opposite of (cosine) similarity

    diagonal = errors.diagonal()

    cost_s = tensor.maximum(0, margin - errors + diagonal)  # compare every diagonal score to scores in its column (all contrastive images for each sentence)
    cost_im = tensor.maximum(0, margin - errors + diagonal.reshape((-1, 1)))  # all contrastive sentences for each image

    cost_tot = cost_s + cost_im

    # clear diagonals
    cost_tot = fill_diagonal(cost_tot, 0)

    return cost_tot.sum()


def encode_sentences(tparams, options, x, mask):
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # Word embedding (source)
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

    # Encode sentences (source)
    proj = get_layer(options['encoder'])[1](tparams, emb, None, options,
                                            prefix='encoder',
                                            mask=mask)
    s = l2norm(proj[0][-1])
    if options['abs']:
        s = abs(s)

    return s

def encode_images(tparams, options, im):
    im_emb = get_layer('ff')[1](tparams, im, options, prefix='ff_image', activ='linear')
    im_emb = l2norm(im_emb)
    if options['abs']:
        im_emb = abs(im_emb)

    return im_emb




def build_model(tparams, options):
    """
    Computation graph for the entire model
    """
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')
    im = tensor.matrix('im', dtype='float32')

    # embed sentences and images
    s_emb = encode_sentences(tparams, options, x, mask)
    im_emb = encode_images(tparams, options, im)

    # Compute loss
    cost = contrastive_loss(s_emb, im_emb, options)

    return [x, mask, im], cost



def build_sentence_encoder(tparams, options):
    """
    Encoder only, for sentences
    """
    # sentence features
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')

    return [x, mask], encode_sentences(tparams, options, x, mask)

def build_image_encoder(tparams, options):
    """
    Encoder only, for images
    """
    # image features
    im = tensor.matrix('im', dtype='float32')
    
    return [im], encode_images(tparams, options, im)


def build_errors(options):
    """ Given sentence and image embeddings, compute the error matrix """
    # input features
    s_emb = tensor.matrix('s_emb', dtype='float32')
    im_emb = tensor.matrix('im_emb', dtype='float32')

    errs = None
    if options['method'] == 'order':
        # trick to make Theano not optimize this into a single matrix op, and overflow memory
        indices = tensor.arange(s_emb.shape[0])
        errs, _ = theano.map(lambda i, s, im: order_violations(s[i], im, options).sum(axis=1).flatten(),
                             sequences=[indices],
                             non_sequences=[s_emb, im_emb])
    else:
        errs = - tensor.dot(s_emb, im_emb.T)

    return [s_emb, im_emb], errs



