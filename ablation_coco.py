import train
dim = 1024
eps = 0
dim_word = 300
norm = 2

base = ['hierarchy', 0.05, True, 'raw', 'relu_oversample']
crop1 = ['hierarchy', 0.05, True, 'raw', 'relu']
symmetric = ['cosine', 0.2, False, 'raw', 'relu_oversample']

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model', choices=['base', 'crop1', 'symmetric'])
args = parser.parse_args()



for method, margin, abs, captions, cnn in [eval(args.model)]:
    name = '_'.join(['coco_new_ablation', method, captions, cnn])
    train.trainer(data='coco', margin=margin, saveto='snapshots/' + name + '.npz', batch_size=128, lrate=0.001, eps=eps, dim_word=dim_word, dim=dim, max_epochs=100, validFreq=300, name=name, norm=norm, abs=abs, cnn=cnn, method=method, captions=captions)
