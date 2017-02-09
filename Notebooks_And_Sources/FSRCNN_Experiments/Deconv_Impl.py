from collections import Iterable

import numpy

import theano.tensor as T

from theano.sandbox.cuda.basic_ops import gpu_contiguous



def compute_tconv_out_size(input_size, filter_size, stride, pad):

    """Computes the length of the output of a transposed convolution



    Parameters

    ----------

    input_size : int, Iterable or Theano tensor

        The size of the input of the transposed convolution

    filter_size : int, Iterable or Theano tensor

        The size of the filter

    stride : int, Iterable or Theano tensor

        The stride of the transposed convolution

    pad : int, Iterable, Theano tensor or string

        The padding of the transposed convolution

    """

    if input_size is None:

        return None

    input_size = numpy.array(input_size)

    filter_size = numpy.array(filter_size)

    stride = numpy.array(stride)



    if isinstance(pad, (int, Iterable)) and not isinstance(pad, str):

        pad = numpy.array(pad)  # to deal with iterables in one line

        output_size = (input_size - 1) * stride + filter_size - 2*pad

    elif pad == 'full':

        output_size = input_size * stride - filter_size - stride + 2

    elif pad == 'valid':

        output_size = (input_size - 1) * stride + filter_size

    elif pad == 'same':

        output_size = input_size

    return output_size


class De_Conv_Layer_ReLU3(object):
    def __init__(self, rng, input, filter_shape, image_shape):
        self.input = input
        print('decon layer....\n')
        print('filter_shape...:', filter_shape)
        print('image_shape...:', image_shape)
        print('input shape....:', input.shape)

        # initialize weights with random weights
        #init = np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]) / np.sqrt(np.prod(image_shape[2])/2)
        #Init from paper
        init = np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]) * .001

        self.W = theano.shared(np.asarray(init,dtype=theano.config.floatX),borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        '''       
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode='full'
        )
        '''
        input_shape = input.shape
        
        #conv_out = theano.tensor.nnet.abstract_conv.conv2d_grad_wrt_inputs(input, self.W, (100,3, 8,8), filter_shape=filter_shape, border_mode='valid', subsample=(2, 2), filter_flip=True)
        conv_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(

            output_grad=input,

            filters=filters,

            input_shape=out_shape,

            filter_shape=kshp,

            border_mode=self.pad,

            subsample=self.stride,

            filter_flip=self.flip_filters)
        #conv_out = theano.tensor.nnet.abstract_conv.conv2d_grad_wrt_inputs(input, self.W, input_shape=out_shape, filter_shape=filter_shape, border_mode=self.pad, subsample=self.stride, filter_flip=self.flip_filters) 
        
        self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input