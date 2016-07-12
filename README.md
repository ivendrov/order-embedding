# order-embeddings

Theano implementation of caption-image retrieval from the paper ["Order-Embeddings of Images and Language"](http://arxiv.org/abs/1511.06361).

(If you're looking for the other experiments, the textual entailment code is in a [separate branch]( https://github.com/ivendrov/order-embedding/tree/textual_entailment), and the hypernym code is [here](https://github.com/ivendrov/order-embeddings-wordnet))

Similar to [visual-semantic-embedding](https://github.com/ryankiros/visual-semantic-embedding) of which this repository is a fork, 
we map images and their captions into a common vector space. The main difference, as explained in the paper, is that we model
the caption-image relationship as an (asymmetric) partial order rather than a symmetric similarity relation.

The code differs from visual-semantic-embedding in a number of other ways, including using 10-crop averaged VGG features for the 
image representation, and adding a visualization server.


## Dependencies

This code is written in python. To use it you will need:

* Python 2.7
* Theano 0.7
* A recent version of [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/)

## Replicating the paper

### Getting data

Download the dataset files (1 GB), including 10-crop [VGG19 features](http://www.robots.ox.ac.uk/~vgg/research/very_deep/), by running

    wget http://www.cs.toronto.edu/~vendrov/order/coco.zip
   
Note that we use the [splits](http://cs.stanford.edu/people/karpathy/deepimagesent/) produced by Andrej Karpathy. The full COCO dataset
can be obtained [here](http://mscoco.org/).
    
Unzip the downloaded file - if not in the project directory, you'll need to change the `datasets_dir` variable in `paths.py`.

**note for Toronto users**: just run `ln -s /ais/gobi1/vendrov/order/coco data/coco` instead
    
### Evaluating pre-trained models

Download two pre-trained models (the full model and the symmetric baseline, 124 MB) and associated visualization data by running

    wget http://www.cs.toronto.edu/~vendrov/order/models.zip
    
Unzip the file in the project directory, and evaluate by running 

```python

    import tools, evaluation
    model = tools.load_model('snapshots/order')
    evaluation.ranking_eval_5fold(model, split='test')
```
    

## Computing image and sentence vectors

Suppose you have a list of strings that you would like to embed into the learned vector space. To embed them, run the following:

    sentence_vectors = tools.encode_sentences(model, s, verbose=True)
    
Where `s` is the list of strings. Note that the strings should already be pre-tokenized, so that `str.split()` returns the tokens.

As the vectors are being computed, it will print some numbers. The code works by extracting vectors in batches of sentences that have the same length - so the number corresponds to the current length being processed. If you want to turn this off, set verbose=False when calling encode.

To encode images, run the following instead:

    image_vectors = tools.encode_images(model, im)
    
Where `im` is a NumPy array of VGG features. Note that the VGG features were scaled to unit norm prior to training the models.

## Training new models

To train your own models, simply run 

    import train
    train.trainer(**kwargs)

As the model trains, it will periodically evaluate on the development set and re-save the model each time performance on the development set increases. Once the models are saved, you can load and evaluate them in the same way as the pre-trained models.

`train.trainer` has many hyperparameters; see `driver.py` for the ones used in the paper. Descriptions of each hyperparameter follow:


#### Saving / Loading
* **name**: a string describing the model, used for saving + visualization
* **save_dir**: the location to save model snapshots
* **load_from**: location of model from which to load existing parameters
* **dispFreq**: How often to display training progress (in batches)
* **validFreq**: How often to evaluate on the development set

#### Data
* **data**: The dataset to train on (currently only 'coco' is supported)
* **cnn**: The name of the CNN features to use, if you want to evaluate different image features

#### Architecture
* **dim**: The dimensionality of the learned embedding space (also the size of the RNN state)
* **dim_image**: The dimensionality of the image features. This will be 4096 for VGG
* **dim_word**: The dimensionality of the learned word embeddings
* **encoder**: The type of RNN to use to encode sentences (currently only 'gru' is supported)
* **margin**: The margin used for computing the pairwise ranking loss

#### Training
* **optimizer**: The optimization method to use (currently only 'adam' is supported)
* **batch_size**: The size of a minibatch.
* **max_epochs**: The number of epochs used for training
* **lrate**: Learning rate
* **grad_clip**: Magnitude at which to clip the gradient

    
## Training on different datasets

To train on a different dataset, put tokenized sentences and image features in the same format as those provided for COCO,
add the relevant paths to `paths.py`, and modify `datasets.py` to handle your dataset correctly.

If you're training on Flickr8k or Flickr30k, just put [Karpathy's](http://cs.stanford.edu/people/karpathy/deepimagesent/) `dataset_flickr{8,30}k.json` file in the dataset directory, and run the scripts `generate_captions.py` and `extract_cnn_features.py`. The latter script requires a working [Caffe installation](http://caffe.berkeleyvision.org/installation.html), as well as the VGG19 [model spec and weights](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77).

The evaluation (`evaluation.py`) and batching (`datasource.py`) assume that there are exactly 5 captions per image; if your dataset doesn't have this property, you will need to modify them.

## Visualizations

You can view plots of training errors and ranking metrics, as well as ROC curves for Image Retrieval, by running the visualization server.
See the `vis` directory for more details.

## Reference

If you found this code useful, please cite the following paper:

Ivan Vendrov, Ryan Kiros, Sanja Fidler, Raquel Urtasun. **"Order-Embeddings of Images and Language."** *arXiv preprint arXiv:1511.06361 (2015).*

    @article{vendrov2015order,
      title={Order-embeddings of images and language},
      author={Vendrov, Ivan and Kiros, Ryan and Fidler, Sanja and Urtasun, Raquel},
      journal={arXiv preprint arXiv:1511.06361},
      year={2015}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)




