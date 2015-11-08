import caffe
import json
import numpy
from collections import defaultdict
import sklearn.preprocessing
from PIL import ImageFile
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True  # needed for coco train



image_loc = {
    'f30k': '/ais/gobi3/u/rkiros/flickr30k/images/',
    'coco': '/ais/gobi3/datasets/mscoco/images/'
}


nets = {
    'VGG19':
            {
                'prototxt': '/u/vendrov/qanda/caffe_models/VGG19/VGG_ILSVRC_19_layers_deploy.prototxt',
                'caffemodel': '/ais/gobi3/datasets/caffe_nets/models/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.caffemodel',
                'features_layer': 'fc7',
                'mean': numpy.array([103.939, 116.779, 123.68])  # BGR means, from https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
            }
}

root_dir = '/u/vendrov/qanda/hierarchy/'


def process_dataset(dataset, net, gpu_id):
    data_dir = root_dir + dataset + '/'
    data = json.load(open(data_dir + 'dataset_%s.json' % dataset, 'r'))

    splits = defaultdict(list)
    for im in data['images']:
        split = im['split']
        if split == 'restval':
            split = 'train'
        splits[split].append(image_loc[dataset] + im['filepath'] + '/' + im['filename'])

    for name, filenames in splits.items():
        run(dataset + '_' + name, filenames, net, gpu_id, data_dir + 'images/')








def run(split_name, filenames, net, gpu_id, output_dir):
    """ Extracts CNN features
    :param split_name: name of the split to use
    :param filenames: list of filenames for images
    :param net: name of the CNN to extract features with
    :param output_dir: the directory to store the features in
    :param gpu_id: gpu ID to use to run computation
    """
    net_data = nets[net]
    layer = net_data['features_layer']

    # load caffe net
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(net_data['prototxt'], net_data['caffemodel'], caffe.TEST)
    batchsize, num_channels, width, height = net.blobs['data'].data.shape

    # set up pre-processor
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_mean('data', net_data['mean'])
    transformer.set_raw_scale('data', 255)



    feat_shape = [len(filenames)] + list(net.blobs[layer].data.shape[1:])
    print("Shape of features to be computed: " + str(feat_shape))

    feats = {}
    for key in ['relu', 'relu_oversample']:
        feats[key] = numpy.zeros(feat_shape).astype('float32')


    for k in range(len(filenames)):
        print('Image %i/%i' % (k, len(filenames)))
        im = caffe.io.load_image(filenames[k])
        h, w, _ = im.shape
        if h < w:
            im = caffe.io.resize_image(im, (256, 256*w/h))
        else:
             im = caffe.io.resize_image(im, (256*h/w, 256))

        crops = caffe.io.oversample([im], (width, height))

        for i, crop in enumerate(crops):
            net.blobs['data'].data[i] = transformer.preprocess('data', crop)

        n = len(crops)

        net.forward()

        output = net.blobs[layer].data[:n]

        for key, f in feats.items():
            if key.find('relu') > -1:
                output = numpy.maximum(output, 0)
            if key.find('oversample') > -1:
                f[k] = output.mean(axis=0)  # mean over 10 crops
            else:
                f[k] = output[4]  # just center crop


    print("Saving features...")
    for methodname, f in feats.items():
        f = sklearn.preprocessing.normalize(f)
        print(methodname)
        method_dir = output_dir + methodname
        try:
            os.mkdir(method_dir)
        except OSError:
            pass

        numpy.save(method_dir + '/%s.npy' % split_name, f)















