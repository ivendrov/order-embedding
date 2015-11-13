import nltk
from nltk.tokenize import word_tokenize
src = '/ais/gobi3/u/rkiros/snli/snli_1.0/snli_1.0_'
dest = '/u/vendrov/qanda/hierarchy/snli/'


def preprocess(cap):
    return ' '.join(word_tokenize(cap))

for split in ['train', 'dev', 'test']:
    with open(src + split + '.txt', 'r') as f:
        f.next()  # skip first line
        with open(dest + split + '.txt', 'w') as dest_f:
            for line in f:
                parts = line.split('\t')

                dest_f.write('\t'.join([parts[0], preprocess(parts[5]), preprocess(parts[6])]) + '\n')
