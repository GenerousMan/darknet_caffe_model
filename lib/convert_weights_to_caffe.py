#-*-coding:utf-8-*-
"""
Created on Tue Jun 6 15:56:31 2018

@author: yangs
"""
from __future__ import print_function

import os
import sys
import argparse
import caffe
import numpy as np

if __name__ == '__main__':

    model_name = 'darknet19'
    darknet_weight_file = './darknet/weights/darknetref.weights'
    caffe_weight_file = './caffe/model/darknetref.caffemodel'
    caffe_model_file = './caffe/proto/darknetref_deploy.prototxt'

    net = caffe.Net(caffe_model_file, caffe.TEST)
    #net.forward()
    for layer_name, blob in net.blobs.iteritems():
        print('{}\t{}'.format(layer_name, blob.data.shape))
    count = 0
    for layer_name, param in net.params.iteritems():
        print('{}\t'.format(layer_name))
        for i in range(len(param)):
            print('{}\t'.format(param[i].data.shape))
            count += np.prod(param[i].data.shape)
        #if 'scale' in layer_name:
            #print("count=", count)
    print('count = {}'.format(count))
    params = net.params.keys()

    # the frist 4 entries are major, minor, revision and net.seen
    netWeightsFloat = np.fromfile(darknet_weight_file, dtype=np.float32)
    netWeights = netWeightsFloat[4:]

    print(netWeights.shape)
    count = 0
    for pr in params:
        print(pr)
        lidx = list(net._layer_names).index(pr)
        layer = net.layers[lidx]
        if count == netWeights.shape[0]:
            print("WARNING: no weigths left for %s" % pr)
            break
        if layer.type == 'Convolution':
            # bias
            if len(net.params[pr]) > 1:
                bias_dim = net.params[pr][1].data.shape
            else:
                bias_dim = (net.params[pr][0].data.shape[0], )
            biasSize = np.prod(bias_dim)
            conv_bias = np.reshape(netWeights[count: count + biasSize], bias_dim)
            if len(net.params[pr]) > 1:
                #assert bias_dim == net.params[pr][1].data.shape
                net.params[pr][1].data[...] = conv_bias
                conv_bias = None
            count += biasSize
            # batch_norm
            next_layer = net.layers[lidx + 1]
            if next_layer.type == 'BatchNorm':
                bn_dims = (3, net.params[pr][0].data.shape[0])
                bnSize = np.prod(bn_dims)
                batch_norm = np.reshape(netWeights[count:count + bnSize], bn_dims)
                count += bnSize
            # weights
            dims = net.params[pr][0].data.shape
            weightSize = np.prod(dims)
            net.params[pr][0].data[...] = np.reshape(netWeights[count:count+weightSize], dims)
            count += weightSize
        elif layer.type == 'BatchNorm':
            net.params[pr][0].data[...] = batch_norm[1] # mean
            net.params[pr][1].data[...] = batch_norm[2] # variance
            net.params[pr][2].data[...] = 1.0 # scale factor
        elif layer.type == 'Scale':
            net.params[pr][0].data[...] = batch_norm[0] # scale
            batch_norm = None
            if len(net.params[pr]) > 1:
                net.params[pr][1].data[...] = conv_bias #bias
                conv_bias = None
            #print('scale\t{}'.format(count))
        else:
            print("WARNING： unsupported layer, " + pr)

    if np.prod(netWeights.shape) != count:
        print("ERROR: size mismatch: %d"%count)
    else:
        print("Convert successfully!")
        net.save(caffe_weight_file)