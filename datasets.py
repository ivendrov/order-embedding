"""
Dataset loading
"""
import numpy
import nltk
from nltk import word_tokenize

def load_dataset(name='snli', load_train=True):
    loc = 'snli/' 

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

