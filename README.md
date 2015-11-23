# order-embeddings

Theano implementation of caption-image retrieval and textual entailment from the paper ["Order-Embeddings of Images and Language"](http://arxiv.org/abs/1511.06361)

[](Describe the ideas of the paper)


## Visualization
[](add links to ranking comparison)

## Dependencies

This code is written in python. To use it you will need:

* Python 2.7
* Theano 0.7
* A recent version of [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/)
* TODO others?

## Replicating the paper

### Getting data

Download the dataset files, including 10-crop [VGG19 features](http://www.robots.ox.ac.uk/~vgg/research/very_deep/), by running

    wget http://www.cs.toronto.edu/~vendrov/datasets/coco.zip
   
Note that we use the splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). The full COCO dataset
can be obtained [here](http://mscoco.org/).
    
In `paths.py`, point `datasets_dir` to where you unzipped the data.
    
### Evaluating pre-trained models

Download the pre-trained models used in the paper by running 

    wget http://www.cs.toronto.edu/~vendrov/datasets/order-models.zip
    




    

## Computing image and sentence vectors

Suppose you have a list of strings that you would like to embed into the learned vector space. To embed them, run the following:

    sentence_vectors = tools.encode_sentences(model, X, verbose=True)
    
Where 'X' is the list of strings. Note that the strings should already be pre-tokenized, so that split() returns the tokens.

As the vectors are being computed, it will print some numbers. The code works by extracting vectors in batches of sentences that have the same length - so the number corresponds to the current length being processed. If you want to turn this off, set verbose=False when calling encode.

To encode images, run the following instead:

    image_vectors = tools.encode_images(model, IM)
    
Where 'IM' is a NumPy array of VGG features. Note that the VGG features were scaled to unit norm prior to training the models.

## Training new models

Open `train.py` and specify the hyperparameters that you would like. Below we describe each of them in detail:

* data: The dataset to train on (f8k, f30k or coco).
* margin: The margin used for computing the pairwise ranking loss. Should be between 0 and 1.
* dim: The dimensionality of the learned embedding space (also the size of the RNN state).
* dim_image: The dimensionality of the image features. This will be 4096 for VGG.
* dim_word: The dimensionality of the learned word embeddings.
* ncon: The number of contrastive (negative) examples for computing the loss.
* encoder: The type of RNN to use. Only supports gru at the moment.
* max_epochs: The number of epochs used for training.
* dispFreq: How often to display training progress.
* decay_c: The weight decay hyperparameter.
* grad_clip: When to clip the gradient.
* maxlen_w: Sentences longer then this value will be ignored.
* optimizer: The optimization method to use. Only supports 'adam' at the moment.
* batch_size: The size of a minibatch.
* saveto: The location to save the model.
* validFreq: How often to evaluate on the development set.
* reload_: Whether to reload a previously trained model.

Once you are happy, just run the following:

    import train
    train.trainer()
    
As the model trains, it will periodically evaluate on the development set (validFreq) and re-save the model each time performance on the development set increases. Generally you shouldn't need more than 15-20 epochs of training on any of the datasets. Once the models are saved, you can load and evaluate them in the same way as the pre-trained models.

## Training on different datasets


## Exploring Regularities


## Reference

If you found this code useful, please cite the following paper:

Ivan Vendrov Ryan Kiros, Sanja Fidler, Raquel Urtasun. **"Order-Embeddings of Images and Language."** *arXiv preprint arXiv:1511.06361 (2015).*

    @article{vendrov2015order,
      title={Order-embeddings of images and language},
      author={Vendrov, Ivan and Kiros, Ryan and Fidler, Sanja and Urtasun, Raquel},
      journal={arXiv preprint arXiv:1511.06361},
      year={2015}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)




