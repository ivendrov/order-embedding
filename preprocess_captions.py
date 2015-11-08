"""
Preprocesses captions, with all instances of 'a', 'the', and  non-alphanumeric tokens removed,
    remaining tokens converted to lower case and lemmatized.
"""
from collections import defaultdict
import json
import nltk
from nltk.corpus import wordnet
import re
import os

lemma_cache = {}

def lemmatize(token):
    if token not in lemma_cache:
        # pick minimum length lemma among all possible POS tags of the word
        options = wordnet._morphy(token, 'n') + wordnet._morphy(token, 'v') + wordnet._morphy(token, 'a') + [token]
        lemma_cache[token] = min(options, key=len)

    #print(token, lemma_cache[token])
    return lemma_cache[token]


def preprocess(tokens):
    # filter out non-alphanumeric tokens, and convert to lowercase
    tokens = [t.lower() for t in tokens if t.isalnum() and t not in ['a', 'the']]
    # lemmatize
    lemmas = map(lemmatize, tokens)
    return lemmas




root_dir = '/u/vendrov/qanda/hierarchy/'

def process_dataset(dataset):
    data_dir = root_dir + dataset
    data = json.load(open('%s/dataset_%s.json' % (data_dir, dataset), 'r'))

    splits = defaultdict(list)
    for im in data['images']:
        split = im['split']
        if split == 'restval':
            split = 'train'

        for s in im['sentences']:
            splits[split].append(s['tokens'])


    for method in ['processed', 'raw']:
        method_dir = os.path.join(data_dir, 'captions', method)

        # create directory if it doesn't already exist
        try:
            os.makedirs(method_dir)
        except OSError:
            pass

        for name, split in splits.items():
            processed_split = map(preprocess, split) if method == 'processed' else split
            with open(os.path.join(method_dir, name + '.txt'), 'w') as f:
                for tokens in processed_split:
                    f.write(' '.join(tokens) + '\n')


