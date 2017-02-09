from __future__ import print_function

import os

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import sys
import scipy.io
import tarfile

import numpy
import timeit
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
import scipy

from inspect import getsourcefile
from os.path import abspath

import matplotlib.pyplot as plt
from theano.tensor.signal import pool

#from PIL import Image
#from scipy.misc import toimage

import timeit
import inspect

from SRCNN_Layers import Conv_Layer_VGG
import scipy.io

def load_weights(mat,layer):
    
    print('Loading weights for Layer: ',layer) 
    
    L=mat['Layers'] #1x37
    this_layer=L[0,layer-1]#1x1
    this_layer_params=this_layer[0,0]#void 
    
    weights=this_layer_params['weights'];#1x2
    
    W=weights[0,0]#3,3,3,64 -> rows,cols,input_feature_size,num_filters
    W=numpy.reshape(W,(W.shape[3],W.shape[2],W.shape[0],W.shape[1])); #num_filters, input_feature_size, rows, cols
    
    B=weights[0,1].astype(theano.config.floatX)#64,1
    B=numpy.reshape(B,(B.shape[0],))
    
    print('Weights loaded with dims: ',W.shape, 'and' ,B.shape) 
    
    return (W,B)


def build_vgg(low_res_batches, high_res_batches, batch_size=500):
    
    mat = scipy.io.loadmat('/Users/himaniarora/Desktop/Columbia/Courses/Neural_networks/project/VGG/matlab/Layers1to7_16vgg.mat')
    
    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    
    im_dims=numpy.asarray([21,21],dtype=int)
    
    print('... building the model')
    print('im_dims=', im_dims)

    # Reshape matrix of rasterized images of shape (batch_size, im_dims[0]*im_dims[1]*3)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer1_input = x.reshape((batch_size, 3, im_dims[0],  im_dims[1]))

    #conv3-64 layers
    layer1 = Conv_Layer_VGG(
        input=layer1_input,
        image_shape=(batch_size, 3, im_dims[0],  im_dims[1]),
        filter_shape=(64, 3, 3, 3),
        weights=load_weights(mat,1)
    )
    
    layer2 = Conv_Layer_VGG(
        input=layer1.output,
        image_shape=(batch_size, 64, im_dims[0],  im_dims[1]),
        filter_shape=(64, 64, 3, 3),
        weights=load_weights(mat,2)
    )  
    
    layer3_input = pool.pool_2d(
        input=layer2.output,
        ds=(2, 2),
        ignore_border=True
    )
    
    im_dims=numpy.asarray(im_dims/2,dtype=int)
    
    #conv3-128 layers
    layer3 = Conv_Layer_VGG(
        input=layer3_input,
        image_shape=(batch_size, 64, im_dims[0],  im_dims[1]),
        filter_shape=(128, 64, 3, 3),
        weights=load_weights(mat,3)
    )
    
    layer4 = Conv_Layer_VGG(
        input=layer3.output,
        image_shape=(batch_size, 128, im_dims[0],  im_dims[1]),
        filter_shape=(128, 128, 3, 3),
        weights=load_weights(mat,4)
    )
    
    layer5_input = pool.pool_2d(
        input=layer4.output,
        ds=(2, 2),
        ignore_border=True
    )
    
    im_dims=numpy.asarray(im_dims/2,dtype=int)
    
    #conv3-256 layers
    layer5 = Conv_Layer_VGG(
        input=layer5_input,
        image_shape=(batch_size, 128, im_dims[0],  im_dims[1]),
        filter_shape=(256, 128, 3, 3),
        weights=load_weights(mat,5)
    )
    
    layer6 = Conv_Layer_VGG(
        input=layer5.output,
        image_shape=(batch_size, 256, im_dims[0],  im_dims[1]),
        filter_shape=(256, 256, 3, 3),
        weights=load_weights(mat,6)
    )
    
    layer7 = Conv_Layer_VGG(
        input=layer6.output,
        image_shape=(batch_size, 256, im_dims[0],  im_dims[1]),
        filter_shape=(256, 256, 3, 3),
        weights=load_weights(mat,7)
    )
    
    y = pool.pool_2d(
        input=layer7.output,
        ds=(2, 2),
        ignore_border=True
    )
    
    im_dims=numpy.asarray(im_dims/2,dtype=int)

    # create a function to compute the output
    test_model = theano.function(
        [x],
        y,
        on_unused_input='ignore'
    )
    
    low_res_output=test_model(low_res_batches)
    high_res_output=test_model(high_res_batches)
    
    #low_res_dims=low_res_output.shape[2]
    #high_res_dims=high_res_output.shape[2]
    
    #grab center pixels
    #center_start = (high_res_dims - low_res_dims) / 2
    #center_end = high_res_dims - center_start
    #sub_high_res_output = high_res_output[:,:,center_start:center_end,center_start:center_end]

    #MSE between center pixels of low resolution and ground truth
    error=T.mean((high_res_output-low_res_output) ** 2)
 
    get_error=theano.function([],error)
    
    return get_error()

    