#!/usr/bin/env python
from pylearn2.utils import serial
from theano import tensor as T
from theano import function
from pylearn2.datasets import cifar10
import numpy as np
import csv


def process(mdl, ds, batch_size=100):
    # This batch size must be evenly divisible into number of total samples!
    mdl.set_batch_size(batch_size)
    X = mdl.get_input_space().make_batch_theano()
    Y = mdl.fprop(X)
    y = T.argmax(Y, axis=1)
    f = function([X], y)
    yhat = []
    for i in xrange(ds.X.shape[0] / batch_size):
        x_arg = ds.X[i * batch_size:(i + 1) * batch_size, :]
        x_arg = ds.get_topological_view(x_arg)
        yhat.append(f(x_arg.astype(X.dtype)))
    return np.array(yhat)


preprocessor = serial.load('kaggle_cifar10_preprocessor.pkl')
mdl = serial.load('kaggle_cifar10_maxout_zca.pkl')
fname = 'kaggle_cifar10_results.csv'
test_size = 100000
sets = 300000 / test_size
res = np.zeros((sets, test_size), dtype='float32')
for n, i in enumerate([test_size * x for x in range(sets)]):
    ds = cifar10.cifar10('test',
                         one_hot=True,
                         toronto_prepro=False,
                         axes=('c', 0, 1, 'b'))
    ds.apply_preprocessor(preprocessor=preprocessor, can_fit=False)
    yhat = process(mdl, ds)
    print "start_idx", i
    res[n, :] = yhat.ravel()

converted_results = [['id', 'label']] + [[n + 1, ds.label_names[int(x)]]
                                         for n, x in enumerate(res.ravel())]
with open(fname, 'w') as f:
    csv_f = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONE)
    csv_f.writerows(converted_results)
