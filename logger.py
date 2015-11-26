"""
Defines a Log class, which is used for outputting statistics while a model is training, which
can then be visualized using vis/visualize_training.html.

The code below is slight modification of https://github.com/ivendrov/torch-logger, see that for detailed documentation.
"""

import json
import os
import os.path as osp


def write_json(file, d):
    name = file + '.json'
    with open(name, 'w') as f:
        json.dump(d, f)


def load_json(file):
    name = file + '.json'
    if not osp.exists(name):
        return None

    with open(name, 'r') as f:
        return json.load(f)

class Log:
    def __init__(self, name, hyperparams, saveDir, xLabel = "Iterations", saveFrequency = 0):
        self.name = name
        self.hyperparams = hyperparams
        self.xLabel = xLabel
        self.saveLoc = osp.join(saveDir, name)
        self.saveFrequency = saveFrequency
        self.data = {}
        self.updatesCounter = 0

        if not osp.exists(self.saveLoc + '.json'):
            write_json(self.saveLoc, [])

        # update index file
        indexLoc = osp.join(saveDir, 'index')
        models = [filename[:-5] for filename in os.listdir(saveDir)
                  if filename.endswith('json') and filename != "index.json"]

        write_json(indexLoc, models)


    def update(self, ys, x=None):
        """ adds the data point (x, ys), where ys is a dictionary of different statistics to keep track of """
        if x is None:
            x = self.updatesCounter

        for name, y in ys.iteritems():
            point = {'x': x, 'y': y}
            # if dataset doesn't exist, create it
            if name not in self.data:
                self.data[name] = []
            # add the point
            self.data[name].append(point)

        self.updatesCounter += 1
        if self.saveFrequency > 0 and self.updatesCounter % self.saveFrequency == 0:
            self.save()

    def save(self, stats=None):
        """ Saves all the data as saveDir/name.json, along with the given statistics """
        write_json(self.saveLoc, {
            'name': self.name,
            'xLabel': self.xLabel,
            'hyperparams': self.hyperparams,
            'data': self.data,
            'stats': stats
        })

