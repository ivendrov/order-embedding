"""
Dataset loading
"""
import numpy

#-----------------------------------------------------------------------------#
# Specify dataset(s) location here
#-----------------------------------------------------------------------------#
def path_to_data(name):
    if name in ['f8k', 'f30k', 'coco']:
        return '/ais/gobi3/u/rkiros/uvsdata/'
    else:
        return '/u/vendrov/qanda/hierarchy/'
#-----------------------------------------------------------------------------#

def load_dataset(name='f8k', load_train=True):
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
            ims = numpy.load(loc+name+'_' + split + '_ims.npy')
        except IOError:
            pass

        edges = []
        try:
            with open(loc+name+'_' + split + '_edges.txt', 'rb') as f:
                for line in f:
                    edges.append(map(int, line.split()))
        except IOError:
            # TODO add edges for image-caption mapping
            pass

        dataset[split] = {'caps': caps, 'edges': edges}
        if ims is not None:
            dataset[split]['ims'] = ims

    return dataset

