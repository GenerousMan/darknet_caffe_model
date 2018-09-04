#-*-coding:utf-8-*-
"""
Created on Tue Jun 6 11:38:55 2018

@author: yangs
"""
from __future__ import print_function

import configparser

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

import utils
import model_libs

if __name__ == "__main__":

    cfg_file = './darknet/cfg/darknet_ref.cfg'
    assert cfg_file.endswith('.cfg'), '{ } is not a .cfg file'.format(cfg_file)

    print('start to parse darknet config file')
    unique_config_file = utils.unique_config_sections(cfg_file)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    # create caffe net
    net = caffe.NetSpec()
    model_name = 'DarknetRef'
    from_layer = 'data'
    phase = 'deploy'
    ntop = 1
    caffe_net_file = './caffe/proto/{}_{}.prototxt'.format(model_name, phase)
    out_label = True
    lr_mult = 1

    transform_param = dict(
        mirror = True,
        crop_size = 224,
        mean_file = 'imagenet_mean.binaryproto',
    )

    data_param = dict(
        source = 'train_lmdb',
        batch_size = 256,
        backend = P.Data.LMDB,
    )

    input_param = dict(
        shape = dict(dim = [1, 3, 227, 227])
    )

    for section in cfg_parser.sections():
        print(section)
        block = dict(cfg_parser.items(section))
        if 'net' in section:
            if phase == 'train' or phase == 'test':
                net.data, net.label = model_libs.DataLayer(name=from_layer, ntop = ntop, phase = phase,
                    transform_param = transform_param, data_param = data_param)
            elif phase == 'deploy':
                net.data = model_libs.InputLayer(name=from_layer, ntop=ntop, input_param = input_param)
        elif 'convolution' in section:
            idx = section.split('_')[1]
            out_layer = '{}{}'.format('conv', idx)
            num_output = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            pad = int(block['pad'])
            # !!Important
            if kernel_size == 1:
                pad = 0
            use_bn = False
            bias_term = True
            if 'batch_normalize' in block:
                use_bn = bool(block['batch_normalize'])
                bias_term = False

            negative_slope = 0.0
            if block['activation'] == 'leaky':
                negative_slope = 0.1 # this value is from paper

            model_libs.ConvBNLayer(net, from_layer, out_layer, use_bn = use_bn, bias_term = bias_term,
                num_output = num_output, kernel_size = kernel_size, pad = pad, stride = stride, 
                lr_mult = lr_mult, negative_slope = negative_slope)
            #model_libs.ConvLayer(net, from_layer, out_layer, num_output = num_output,
            #    pad = pad, kernel_size = kernel_size, stride = stride, bias_term=True)
            from_layer = out_layer
        elif 'maxpool' in section:
            idx = section.split('_')[1]
            out_layer = '{}{}'.format('pool', idx)
            size = int(block['size'])
            stride = int(block['stride'])
            model_libs.PoolingLayer(net, from_layer, out_layer, kernel_size=size, stride=stride)
            from_layer = out_layer
        elif  'avgpool' in section:
            out_layer = 'global_avg_pool'
            model_libs.PoolingLayer(net, from_layer ,out_layer, 
                    pool=P.Pooling.AVE, global_pooling=True)
            from_layer = out_layer
        elif 'softmax' in section:
            out_layer = 'prob'
            model_libs.SoftmaxLayer(net, from_layer, out_layer)
            break
        '''
        for key, val in block.items():
            print(key, val)
        '''

    with open(caffe_net_file, 'w') as f:
        print("name: '{}'".format(model_name), file=f)
        print(net.to_proto(), file=f)