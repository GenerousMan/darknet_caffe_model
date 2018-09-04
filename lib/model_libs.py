#-*-coding:utf-8-*-
"""
Created on Tue Jun 6 13:40:06 2018

@author: yangs
"""
from __future__ import print_function

import utils
import os
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

# phase = 'train' or 'test' or 'deploy'
def DataLayer(name = 'data', ntop=2, phase='train', 
    transform_param={}, data_param={}, input_param = {}):
    assert phase.upper() == "TRAIN" or phase.upper() == "TEST", 'phase type can only be TRAIN or TEST'
    kwargs = {
        'include': dict(phase=caffe_pb2.Phase.Value(phase.upper())),
        'transform_param': transform_param,
        'data_param': data_param,
    }
    return L.Data(name=name, ntop=ntop, **kwargs)
    
def InputLayer(name = 'data', ntop=1, input_param = {}):
    return L.Input(name=name, ntop=ntop, input_param=input_param)

def PoolingLayer(net, from_layer, out_layer, pool = P.Pooling.MAX, 
    kernel_size = 2, stride = 2, global_pooling = True):
    assert pool == P.Pooling.MAX or pool == P.Pooling.AVE, 'Pooling type can only be MAX or AVG'
    if pool == P.Pooling.MAX:
        net[out_layer] = L.Pooling(net[from_layer], pool = pool, 
                        kernel_size = kernel_size, stride = stride)
    else:
        net[out_layer] = L.Pooling(net[from_layer], pool = pool,
                        global_pooling = global_pooling)

def ConvLayer(net, from_layer, out_layer, num_output, pad, kernel_size, 
    stride, bias_term=False, conv_prefix='', conv_postfix='', lr_mult=1):
    
    weight_filler = dict(type='gaussian', std=0.01)
    conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
    [kernel_h, kernel_w] = utils.UnpackVariable(kernel_size, 2)
    [pad_h, pad_w] = utils.UnpackVariable(pad, 2)
    [stride_h, stride_w] = utils.UnpackVariable(stride, 2)

    if kernel_h == kernel_w:
        convolution_param = dict(num_output = num_output, 
                            kernel_size = kernel_h, 
                            pad = pad_h, 
                            stride = stride_h, weight_filler =weight_filler)
    else:
        convolution_param = dict(num_output = num_output, 
                            kernel_h = kernel_h, kernel_w =kernel_w,
                            pad_h = pad_h, pad_w = pad_w,
                            stride_h = stride_h, stride_w = stride_w)

    if bias_term:
        kwargs = {
            'param': [dict(lr_mult = lr_mult, decay_mult = 1),
                        dict(lr_mult = lr_mult * 2, decay_mult = 0)],
            #'param': [dict(lr_mult = lr_mult * 2, decay_mult = 1)],
        }
        bias_filler = dict(type='constant', value = 0.1)
        convolution_param['bias_filler'] = bias_filler
    else:
        kwargs = {
            'param': [dict(lr_mult = lr_mult, decay_mult = 1)],
        }
        convolution_param['bias_term'] = bias_term

    net[out_layer] = L.Convolution(net[from_layer], 
                    convolution_param=convolution_param, **kwargs)

def ConvBNLayer(net, from_layer, out_layer, num_output, kernel_size, pad, stride, 
    use_bn = False, bias_term = False, lr_mult = 1, negative_slope = 0.0, 
    conv_prefix = '', conv_postfix = '', **bn_prams):
    
    if use_bn:
        eps = bn_prams.get('eps', 0.001)
        moving_average_fraction = bn_prams.get('moving_average_fraction', 0.999)
        use_global_stats = bn_prams.get('use_global_stats', False)
        # parameters for batchnorm layer
        bn_kwargs = {
            'eps': eps,
            'moving_average_fraction': moving_average_fraction,
            'use_global_stats': False,
        }

        bn_lr_mult = lr_mult
        if use_global_stats:
            # only specify if use_global_stats is explitiyly provides;
            # otherwise, use_global_stats_ == this->phase_ == TEST;
            bn_kwargs = {
                'eps': eps,
                'use_global_stats': use_global_stats,
            }

            bn_lr_mult = 0

        #if use_scale:
        sb_kwargs = {
            'bias_term':True,
            'param': [
                dict(lr_mult = bn_lr_mult, decay_mult = 0),
                dict(lr_mult = bn_lr_mult, decay_mult = 0)],
                'filler': dict(type='constant', value=1.0),
                'bias_filler': dict(type='constant', value=0.0),
        }

    ConvLayer(net, from_layer, out_layer, num_output=num_output, 
        kernel_size=kernel_size, pad=pad, stride=stride, bias_term=bias_term, 
        conv_prefix=conv_prefix, conv_postfix=conv_postfix, lr_mult= lr_mult)

    conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
    if use_bn:
        bn_name = '{}_bn'.format(conv_name)
        net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)

        #if use_scale:
        sb_name = '{}_scale'.format(conv_name)
        net[sb_name] = L.Scale(net[conv_name], in_place=True, **sb_kwargs)

    #if use_relu:
    relu_name = '{}_relu'.format(conv_name)
    net[relu_name] = L.ReLU(net[conv_name], negative_slope=negative_slope, in_place = True)


def SoftmaxLayer(net, from_layer, out_layer):
    net[out_layer] = L.Softmax(net[from_layer])