# coding: utf-8
import datasource, datasets
d = load_dataset('denotation')
d = datasets.load_dataset('denotation')
dev = datasource.Datasource(d['dev'], {})
train = datasource.Datasource(d['train'], {})
dev_caps, dev_edges, dev_target = dev.all()
train_caps, train_edges, train_target = train.all()
train_set = {}
for i in range(len(train_edges)):
    edge = train_edges[i]
    edge = (edge[0], edge[1])
    train_set[edge] = train_target[i]
    
edges_correct = 0
edges_wrong = 0
train_cap_set = dict(map(reversed, enumerate(dev_caps)))
train_cap_set = dict(map(reversed, enumerate(train_caps)))
nodes_shared = 0
len(cap for cap in dev_caps if cap in train_cap_set)
len([cap for cap in dev_caps if cap in train_cap_set])
len(dev_caps)
len(train_caps)
dev_cap_set = dict(map(reversed, enumerate(dev_caps))
)
for i in range(len(dev_edges)):
    edge = dev_edges[i]
    edge = (edge[0], edge[1])
    caps = map(lambda n: dev_caps[n], edge)
    if caps[0] in train_cap_set and caps[1] in train_cap_set:
        train_edge = map(lambda c: train_cap_set[c], caps)
        if train_edge in train_set:
            target = train_set[train_edge]
            my_target = dev_target[i]
            if target == my_target:
                edges_correct += 1
            else:
                edges_wrong += 1
                
for i in range(len(dev_edges)):
    edge = dev_edges[i]
    edge = (edge[0], edge[1])
    caps = map(lambda n: dev_caps[n], edge)
    if caps[0] in train_cap_set and caps[1] in train_cap_set:
        train_edge = tuple(map(lambda c: train_cap_set[c], caps))
        if train_edge in train_set:
            target = train_set[train_edge]
            my_target = dev_target[i]
            if target == my_target:
                edges_correct += 1
            else:
                edges_wrong += 1
                
edges_correct
edges_wrong
len(edges)
len(dev_edges)
len(dev_caps)
len(train_caps)
len(set(dev_caps))
len(set(train_caps))
len(set(train_caps) & set(dev_caps))
