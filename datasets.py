"""
Dataset loading
"""
import numpy

#-----------------------------------------------------------------------------#
# Specify dataset(s) location here
#-----------------------------------------------------------------------------#
def path_to_data(name):
    return '/u/vendrov/qanda/hierarchy/'
#-----------------------------------------------------------------------------#

def load_dataset(name, cnn, captions, load_train=True):
    """
    Load captions and image features
    Possible options: f8k, f30k, coco
    """
    loc = path_to_data(name) + name 

    splits = []
    if load_train:
        splits = ['train', 'dev', 'test']
    else:
        splits = ['dev', 'test']


    dataset = {}

    for split in splits:
        dataset[split] = {}
        caps = []
        splitName = 'val' if name == 'coco' and split == 'dev' else split
        with open('%s/captions/%s/%s.txt' % (loc, captions, splitName), 'rb') as f:
            for line in f:
                caps.append(line.strip())
            dataset[split]['caps'] = caps

        dataset[split]['ims'] = numpy.load('%s/images/%s/%s.npy' % (loc, cnn, splitName))

        # handle coco specially by only taking the first 1k images / 5k captions
        if name == 'coco' and split in ['dev', 'test']:
            dataset[split]['ims'] = dataset[split]['ims'][:1000]
            dataset[split]['caps'] = dataset[split]['caps'][:5000]

    return dataset

