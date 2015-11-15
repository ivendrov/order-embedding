"""
Dataset loading
"""
import numpy
import nltk
from nltk import word_tokenize

#-----------------------------------------------------------------------------#
# Specify dataset(s) location here
#-----------------------------------------------------------------------------#
def path_to_data(name):
    if name in ['f8k', 'f30k', 'coco']:
        return '/ais/gobi3/u/rkiros/uvsdata/'
    else:
        return '/u/vendrov/qanda/hierarchy/'
#-----------------------------------------------------------------------------#

def load_dataset(name='snli', load_train=True):
    """
    Load captions and image features
    Possible options: f8k, f30k, coco
    """

    loc = '/u/vendrov/qanda/hierarchy/snli/' # '/ais/gobi3/u/rkiros/snli/snli_1.0/snli_1.0_'

    splits = []
    if load_train:
        splits = ['train', 'dev', 'test']
    else:
        splits = ['dev', 'test']



    dataset = {}

    for split in splits:
        caps = []
        labels = []

        with open(loc + split + '.txt') as f:
            for line in f:
                parts = line.strip().split('\t')
                caps.append((parts[1], parts[2]))
                labels.append(parts[0] == 'entailment')


        dataset[split] = {'caps': caps, 'labels': numpy.array(labels).astype('float32')}

    return dataset

