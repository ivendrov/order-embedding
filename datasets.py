"""
Dataset loading
"""
import numpy
import paths

def load_dataset(name, cnn, load_train=True, fold=0):
    """
    Load captions and image features
    Possible options: f8k, f30k, coco
    """
    loc = paths.datasets[name]['loc']

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
        with open('%s/%s.txt' % (loc, splitName), 'rb') as f:
            for line in f:
                caps.append(line.strip())
            dataset[split]['caps'] = caps

        dataset[split]['ims'] = numpy.load('%s/images/%s/%s.npy' % (loc, cnn, splitName))

        # handle coco specially by only taking 1k or 5k captions/images
        if name == 'coco' and split in ['dev', 'test']:
            dataset[split]['ims'] = dataset[split]['ims'][fold*1000:(fold+1)*1000]
            dataset[split]['caps'] = dataset[split]['caps'][fold*5000:(fold+1)*5000]

    return dataset

