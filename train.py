"""
Main trainer function
"""
import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy
import copy

import os
import warnings
import sys
import time

import hierarchy_data

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import *
from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer
from optim import adam
from model import init_params, build_model, build_sentence_encoder, build_image_encoder, build_errors
from vocab import build_dictionary
from evaluation import eval_accuracy
from tools import encode_sentences, encode_images
from datasets import load_dataset

# main trainer
def trainer(data='snli',
            margin=0.2,
            dim=1024,
            dim_image=4096,
            dim_word=300,
            encoder='gru',
            max_epochs=15,
            dispFreq=10,
            decay_c=0.,
            grad_clip=2.,
            maxlen_w=None,
            optimizer='adam',
            batch_size = 128,
            validFreq=100,
            lrate=0.001,
            eps=0,
            norm=1,
            name='anon',
            overfit=False,
            saveto='',
            load_from = None):

    # Model options
    model_options = {}
    model_options['data'] = data
    model_options['margin'] = margin
    model_options['dim'] = dim
    model_options['dim_image'] = dim_image
    model_options['dim_word'] = dim_word
    model_options['encoder'] = encoder
    model_options['max_epochs'] = max_epochs
    model_options['dispFreq'] = dispFreq
    model_options['decay_c'] = decay_c
    model_options['grad_clip'] = grad_clip
    model_options['maxlen_w'] = maxlen_w
    model_options['optimizer'] = optimizer
    model_options['batch_size'] = batch_size
    model_options['saveto'] = saveto
    model_options['validFreq'] = validFreq
    model_options['lrate'] = lrate
    model_options['eps'] = eps
    model_options['norm'] = norm
    model_options['load_from'] = load_from

    saveto = saveto + '/' + name
    import datetime
    timestampedName = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + name

    from logger import Log
    log = Log(name=timestampedName, hyperparams=model_options, saveDir='vis/training/',
              xLabel='Examples Seen', saveFrequency=1)



    print model_options

    # reload options, without overwriting existing ones
    if load_from is not None and os.path.exists(load_from):
        print 'reloading...' + load_from
        with open('%s.pkl'%load_from, 'rb') as f:
            old_model_options = pkl.load(f)
            for k, v in old_model_options.iteritems():
                if k not in model_options:
                    model_options[k] = v



    # Load training and development sets
    print 'Loading dataset'
    dataset = load_dataset(data, load_train=not overfit)
    train = dataset['dev'] if overfit else dataset['train']

    def flattenCaps(caps):
        return [c for cs in caps for c in cs]

    dev = dataset['dev']

    # Create and save dictionary
    print 'Creating dictionary'
    worddict = build_dictionary(flattenCaps(train['caps']) + flattenCaps(dev['caps']))[0]
    n_words = len(worddict)
    model_options['n_words'] = n_words
    print 'Dictionary size: ' + str(n_words)
    with open('%s.dictionary.pkl'%saveto, 'wb') as f:
        pkl.dump(worddict, f)

    # Inverse dictionary
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'


    print 'Loading data'
    # Each sentence in the minibatch have same length (for encoder)
    train_iter = hierarchy_data.HierarchyData(train, batch_size=batch_size, worddict=worddict,
                                              n_words=n_words, maxlen=maxlen_w)
    dev = hierarchy_data.HierarchyData(dev, worddict=worddict, n_words=n_words, maxlen=maxlen_w)
    #test = hierarchy_data.HierarchyData(dataset['test'], worddict=worddict, n_words=n_words)

    dev_caps, dev_target = dev.all()
    #test_caps, test_edges, test_target = dev.all()

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if load_from is not None and os.path.exists(load_from):
        params = load_params(load_from, params)

    tparams = init_tparams(params)

    trng, inps, cost = build_model(tparams, model_options)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=False)
    print 'Done'

    # weight decay, if applicable
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=False)
    print 'Done'

    print 'Building sentence encoder'
    trng, inps_se, sentences = build_sentence_encoder(tparams, model_options)
    f_senc = theano.function(inps_se, sentences, profile=False)

    print 'Building hierarchical error'
    inps_he, errors = build_errors(tparams, model_options)
    h_error = theano.function(inps_he, errors, profile=False)

    print 'Building f_grad...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    f_grad_norm = theano.function(inps, [(g**2).sum() for g in grads], profile=False)
    f_weight_norm = theano.function([], [(t**2).sum() for k,t in tparams.iteritems()], profile=False)

    if grad_clip > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (grad_clip**2),
                                           g / tensor.sqrt(g2) * grad_clip,
                                           g))
        grads = new_grads

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    # (compute gradients), (updates parameters)
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)

    print 'Optimization'

    uidx = 0
    curr = 0.
    n_samples = 0


    
    for eidx in xrange(max_epochs):

        print 'Epoch ', eidx

        for x, mask, labels in train_iter:
            n_samples += x.shape[1]
            uidx += 1

            # Update
            ud_start = time.time()
            cost = f_grad_shared(x, mask, labels)
            f_update(lrate)
            ud = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud
                log.update({'Error': float(cost)}, n_samples)
                # print("Fraction of RNN computation wasted: " + str(1 - mask.mean()))


            if numpy.mod(uidx, validFreq) == 0:

                print 'Computing results...'
                curr_model = {}
                curr_model['options'] = model_options
                curr_model['worddict'] = worddict
                curr_model['word_idict'] = word_idict
                curr_model['f_senc'] = f_senc
                #curr_model['f_ienc'] = f_ienc
                curr_model['h_error'] = h_error

                # encode sentences efficiently
                dev_s = encode_sentences(curr_model, dev_caps, batch_size=batch_size)

                # compute errors
                dev_errs = h_error(dev_s)

                # compute accuracy
                accuracy, wrong_indices, wrong_preds = eval_accuracy(dev_errs, dev_target, dev_errs, dev_target)
                print("Accuracy: %.5f" % accuracy)
                log.update({'Accuracy': accuracy}, n_samples)

                if accuracy > curr:
                    curr = accuracy
                    # Save model
                    print 'Saving...',
                    params = unzip(tparams)
                    numpy.savez(saveto, **params)
                    pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                    print 'Done'

        print 'Seen %d samples'%n_samples

if __name__ == '__main__':
    pass

