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

def load_dataset(name='f8k', load_train=True, cnn=None):
    """
    Load captions and image features
    Possible options: f8k, f30k, coco
    """
    loc = path_to_data(name) + name + '/'

    splits = []
    if load_train:
        splits = ['train', 'dev', 'test']
    else:
        splits = ['dev', 'test']


    dataset = {}

    for split in splits:
        caps = []
        with open(loc+name+'_' + split + '_caps.txt', 'rb') as f:
            for line in f:
                caps.append(line.strip())

        ims = None
        try:
            ims = numpy.load(loc + cnn + '/' + 'f30k_' + split + '_ims.npy')
        except IOError:
            pass

        dataset[split] = {'caps': caps}
        if ims is not None:
            dataset[split]['ims'] = ims

    return dataset

