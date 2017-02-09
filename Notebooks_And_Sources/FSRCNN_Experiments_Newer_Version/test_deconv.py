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

input = numpy.zeros((1,56,8,8 ))
image_shape = input.shape
filter_shape = (3,57,9,9)
filters = numpy.zeros((1,56,8,8))
print(theano.tensor.nnet.abstract_conv.conv2d_grad_wrt_weights(input, filters, filter_shape, image_shape, 'valid', (2,2)))
#print(conv_out(input, filters, filter_shape, image_shape,'valid', (2,2)))
#print(conv_out.shape())
