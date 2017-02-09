"""This code borrows elements from: 

[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
from __future__ import print_function

import sys
import numpy as np
import timeit
import inspect
import numpy
import scipy

from theano.tensor.nnet import conv

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
from theano.tensor.signal import pool

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import Iterable

import numpy

import theano.tensor as T

from theano.sandbox.cuda.basic_ops import gpu_contiguous

class De_Conv_Layer_ReLU(object):
    def __init__(self, rng, input, filter_shape, image_shape):
        self.input = input
        print('decon layer....\n')
        print('filter_shape...:', filter_shape)
        print('image_shape...:', image_shape)
        print('input shape....:', input.shape)

        # initialize weights with random weights
        #init = np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]) / np.sqrt(np.prod(image_shape[2])/2)
        #Init from paper
        #init = np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]) * .001
        init = np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]) / np.sqrt(np.prod(image_shape[2])/2)
        #Init from paper
        #init = np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]) * .001

        
        self.W = theano.shared(np.asarray(init,dtype=theano.config.floatX),borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        self.output_size = (23,23)
        input_shape = input.shape
        self.output_shape = (50,3, 23, 23)
        print(input_shape)
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(

            imshp=self.output_shape,

            kshp=(filter_shape[1],filter_shape[0],filter_shape[2], filter_shape[3]),

            subsample=(2,2), border_mode='valid',

            filter_flip=True)

        output_size = self.output_shape[2:]

        if isinstance(self.output_size, T.Variable):

            output_size = self.output_size

        elif any(s is None for s in output_size):

            output_size = self.get_output_shape_for(input.shape)[2:]

        conved = op(self.W, input, output_size)

        
        self.output = conved + self.b.dimshuffle('x', 0, 'x', 'x')

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
      
