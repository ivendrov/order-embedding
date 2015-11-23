import train

base = ['hierarchy', 0.05, True, '10crop']
crop1 = ['hierarchy', 0.05, True, '1crop']
symmetric = ['cosine', 0.2, False, '10crop']

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model', choices=['base', 'crop1', 'symmetric'])
args = parser.parse_args()


params = {
    'dim_image': 4096,
    'encoder': 'gru',
    'dispFreq': 10,
    'grad_clip': 2.,
    'optimizer': 'adam',
    'batch_size': 128,
    'norm': 2,
    'dim': 1024,
    'dim_word': 300,
    'lrate': 0.001,
    'validFreq': 300
}



for method, margin, abs, cnn in [eval(args.model)]:
    name = '_'.join(['anon', 'coco_new_ablation', method, cnn])
    train.trainer(data='coco', margin=margin, max_epochs=100, name=name, abs=abs, cnn=cnn, method=method, **params)
