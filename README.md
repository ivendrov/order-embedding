# visual-semantic-embedding

Code for the textual entailment experiment (only the order-embedding version, not the baselines) from the paper. 
This branch has not yet been polished, and isn't officially supported (please don't file Github issues about it) - just provided as a convenience.

Same dependencies as master branch (Numpy & Theano).

## Get SNLI
Get our processed version of the SNLI corpus by running:

	wget http://www.cs.toronto.edu/~vendrov/order/snli.zip

and unzipping the downloaded file in the project directory.

## Training
Run
```python

import train
train.trainer(saveto='your_snapshots_directory',name='model_name')
```
(If you want to live to see the program terminate, we recommend you use a GPU by setting the appropriate Theano flags).

## Evaluating
Modify `eval_entailment.py` with the appropriate path to the model snapshot, then run it.
