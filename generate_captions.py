"""
Converts Karpathy's dataset.json files to a plain text file for each split, with 1 caption on each line
"""
from collections import defaultdict
import json

root_dir = '/u/vendrov/qanda/hierarchy/'

def process_dataset(dataset):
    data_dir = root_dir + dataset
    data = json.load(open('%s/dataset_%s.json' % (data_dir, dataset), 'r'))

    splits = defaultdict(list)
    for im in data['images']:
        split = im['split']
        if split == 'restval':
            split = 'train'

        for s in im['sentences'][:5]:  # take exactly 5 captions per image
            splits[split].append(s['tokens'])

    for name, split in splits.items():
        with open(data_dir + '/' + name + '.txt', 'w') as f:
            for tokens in split:
                f.write(' '.join(tokens) + '\n')


