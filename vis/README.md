# Visualization Server

## Running the server
Run `python -m SimpleHTTPServer` in this directory. Then point your browser to the running server (usually at `localhost:8000`)
and open either `visualize_training.html` or `visualize_roc.html`.

Note that modern browsers like Chrome will cache files, making it hard to monitor models as they are training. I use
the [Cache Killer](https://chrome.google.com/webstore/detail/cache-killer/jpfbieopdmepaolggioebjmedmclkbap?hl=en) Chrome extension to
circumvent this.

The visualizations use a number of JSON files stored in the `training` and `roc` directories. 
If you train using COCO and cross-validate on the first 1000 images of the validation set (the default) you don't
need to change anything - the `train.py` script will generate all necessary files for you. If you want to train
on your own data, or if you want to modify the visualizations themselves, the following documentation may prove helpful:

## Training Visualization

#### Expected File Structure

`training/index.json`: list of experiments to be visualized

`training/{experimentName}.json`: stores training trace of the given experiment

#### JSON Schema for training trace

```js
experimentName.json := {
    "data": str -> [point],
    "hyperparams": str -> value
}

point := {
    "x": float, // usually some measure of time or number of examples / batches seen
    "y": float  // value 
}
```

## ROC Visualization

Displays the entire precision-recall curve, allowing a more detailed analysis of which images different methods do better on.

#### Expected File Structure

`roc/index.json`: dictionary mapping each dataset to a list of experiments performed on that dataset

`roc/{dataset}/{split}/image_urls.json` : list of image urls 

`roc/{dataset}/{split}/captions.json` : list of captions

`roc/{dataset}/{split}/{experimentName}.json` : results file for the experiment; contains statistics + top retrieved images for each caption

JSON schemas for all the types of files are listed below:

### JSON schema for index.json:

```js
index.json := {
    "stats":[stat]
    "sentences":[sentence]
}
stat := {
    "name": str
    "value": num
}
sentence := {
    "id": int
    "rank": int
    "gt_image": image
    "top_images": [image]
}

image := {
    "id": int
    "score": float
}
```