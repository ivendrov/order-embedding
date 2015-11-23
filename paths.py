import numpy
datasets = {
    'coco': {
        'dir': '/u/vendrov/qanda/hierarchy/coco',
        'images_dir': '/ais/gobi3/datasets/mscoco/images'
    }
}


cnns = {
    'VGG19':
            {
                'prototxt': '/u/vendrov/qanda/caffe_models/VGG19/VGG_ILSVRC_19_layers_deploy.prototxt',
                'caffemodel': '/ais/gobi3/datasets/caffe_nets/models/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.caffemodel',
                'features_layer': 'fc7',
                'mean': numpy.array([103.939, 116.779, 123.68])  # BGR means, from https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
            }
}