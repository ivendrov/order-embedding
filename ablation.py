import train
dim = 1024
eps = 0
dim_word = 300
norm = 2


#base = ['hierarchy', 0.05, True, 'processed', 'relu_oversample']
crop1 = ['hierarchy', 0.05, True, 'processed', 'relu']
raw = ['hierarchy', 0.05, True, 'raw', 'relu_oversample']
symmetric = ['cosine', 0.2, False, 'processed', 'relu_oversample']


for method, margin, abs, captions, cnn in [symmetric]:
    name = '_'.join(['coco', method, captions, cnn])
    train.trainer(data='coco', margin=margin, saveto='snapshots/' + name + '.npz', batch_size=128, lrate=0.001, eps=eps, dim_word=dim_word, dim=dim, max_epochs=80, validFreq=300, name=name, norm=norm, abs=abs, cnn=cnn, method=method, captions=captions, diagonal_weight=2)
