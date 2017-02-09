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


def translate_image(X):
    lenX = X.shape[0]
    #print X.shape
    iter = 0
    deepX = X[:]
    while iter < lenX:
        #controlling how much data to change
        if(numpy.random.random() > 1):
            iter = iter + 1
            continue
        im=numpy.reshape(deepX[iter],(3,8,8))
        im = im.transpose(1,2,0)
        randx = numpy.random.randint(0,1)
        randy = numpy.random.randint(0,1)
        if(numpy.random.random() > 0.5):
            randx = randx * -1
        if(numpy.random.random() > 0.5):
            randy = randy * -1
        im2 = scipy.ndimage.shift(im,[randx,randy,0])
        deepX[iter] = im2.transpose(2,0,1).flatten()
        iter = iter + 1
    return deepX
def rotate_image(X):
    lenX = X.shape[0]
    #print X.shape
    iter = 0
    deepX = X[:]
    while iter < lenX:
        #controlling how much data to change
        if(numpy.random.random() > 1):
            iter = iter + 1
            continue
        randx = numpy.random.randint(0,5)    
        theta = randx
        if(numpy.random.random() > 0.5):
            theta = theta *-1
        im = numpy.reshape(deepX[iter],(3,8,8))
        im = im.transpose(1,2,0)
        im2 = scipy.ndimage.rotate(im, theta+0.001, reshape=False)
        deepX[iter] = im2.transpose(2,0,1).flatten()
        iter = iter + 1
    return deepX


def noise_image(X, gaussian_noise = True):
    lenX = X.shape[0]
    #print X.shape
    iter = 0
    deepX = X [:]
    while iter < lenX:
        #controlling how much data to change
        if(numpy.random.random() > 1):
            iter = iter + 1
            continue
        randx = numpy.random.randint(0,6)    
        theta = randx
        if(numpy.random.random() > 0.5):
            theta = theta *-1
        im = numpy.reshape(deepX[iter],(3,8,8))
        im = im.transpose(1,2,0)
        im2 = im
        if(gaussian_noise):
            noise = numpy.random.normal(0, 0.0025, [8,8,3])
            im2 = noise + im2
        else:
            noise = numpy.random.uniform(low=-0.0025, high=0.0025, size=[8,8,3]) 
            im2 = im2 + noise
           
        deepX[iter] = im2.transpose(2,0,1).flatten()
        iter = iter + 1
    return deepX
    
#Implement a convolutional neural network with the translation method for augmentation
#def test_lenet_translation():


#Problem 2.2
#Write a function to ad#d roatations
#def rotate_image():
#Implement a convolutional neural network with the rotation method for augmentation
#def test_lenet_rotation():

#Problem 2.3
#Write a function to flip images
def flip_image(X):
    lenX = X.shape[0]
    #print X.shape
    iter = 0
    deepX =X[:]
    while iter < lenX:
        if(numpy.random.random() > 0.5):
            iter = iter + 1
            continue
        temp = numpy.reshape(deepX[iter],(3,8,8)).transpose(1,2,0) 
        deepX[iter] = numpy.fliplr(temp).transpose(2,0,1).flatten()
        iter = iter + 1
    return deepX




#This conv layer reduces output volume by (W - F + 1)
#TODO: Toggle initialization to He init vs. SRCNN's original N(0,.001)
class Conv_Layer_PReLU(object):
    def __init__(self, rng, input, filter_shape, image_shape):
        print('lovely....\n')
        print('filter_shape...:', filter_shape)
        print('image_shape...:', image_shape)
        print('input shape....:', input.shape)
        self.input = input

        # initialize weights with random weights

        ##Xavier-style init
        init = np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]) / np.sqrt(np.prod(image_shape[2])/2)
        #Init from paper
        #init = np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]) * .001

        self.W = theano.shared(np.asarray(init,dtype=theano.config.floatX),borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        
        self.b = theano.shared(value=b_values, borrow=True)
        self.a = theano.shared(np.cast[theano.config.floatX](0.5))
        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode='half'
        )
        #print(a_value.shape)
        self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'), alpha = self.a)

        # store parameters of this layer
        self.params = [self.W, self.b, self.a]

        # keep track of mode
        
class Conv_Layer_ReLU(object):
    def __init__(self, rng, input, filter_shape, image_shape):
        print('lovely....\n')
        print('filter_shape...:', filter_shape)
        print('image_shape...:', image_shape)
        print('input shape....:', input.shape)
        self.input = input

        # initialize weights with random weights

        ##Xavier-style init
        init = np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]) / np.sqrt(np.prod(image_shape[2])/2)
        #Init from paper
        #init = np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]) * .001

        self.W = theano.shared(np.asarray(init,dtype=theano.config.floatX),borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode='half'
        )
        self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

class Conv_Layer_None(object):
    def __init__(self, rng, input, filter_shape, image_shape):
        self.input = input

        # initialize weights with random weights
        #init = np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]) / np.sqrt(np.prod(image_shape[2])/2)
        #Init from paper
        init = np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]) * .001

        self.W = theano.shared(np.asarray(init,dtype=theano.config.floatX),borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode='valid'
        )
        self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

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
        init = np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]) * .001

        self.W = theano.shared(np.asarray(init,dtype=theano.config.floatX),borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
               
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode='full'
        )
        self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
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
class De_Conv_Layer_ReLU2(object):
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
        filters = gpu_contiguous(self.W)

        input = gpu_contiguous(input)
        
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
        
        #input_shape = input.shape
        #out_shape = compute_tconv_out_size(input.shape[2:], filter_shape,

         #                                  (2,2), 'valid')

        #for el in out_shape:

        #    if isinstance(el, T.TensorVariable):

         #       el = None
        #in_shape = input.eval().shape
        #self.num_filters = filter_shape[0]
        #out_shape = [in_shape[0]] + [self.num_filters] + list(out_shape)
        
        #deconv_op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(imshp=,kshp=filter_shape, border_mode='valid', subsample=(2, 2), filter_flip=True)
        conv_out = theano.tensor.nnet.abstract_conv.conv2d_grad_wrt_weights(input, filters, filter_shape, image_shape 'valid')      
        #conv_out = deconv_op(filters, input, out_shape[2:])
        #conv_out = theano.tensor.nnet.abstract_conv.conv2d_grad_wrt_inputs(input, self.W, input_shape=out_shape, filter_shape=filter_shape, border_mode=self.pad, subsample=self.stride, filter_flip=self.flip_filters) 
        
        self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

        
def train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,output_len,decay_learning_rate_function,
            verbose = True):
    """
    Wrapper function for training and test THEANO model

    :type train_model: Theano.function
    :param train_model:

    :type validate_model: Theano.function
    :param validate_model:

    :type test_model: Theano.function
    :param test_model:

    :type n_train_batches: int
    :param n_train_batches: number of training batches

    :type n_valid_batches: int
    :param n_valid_batches: number of validation batches

    :type n_test_batches: int
    :param n_test_batches: number of testing batches

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to

    """

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    decay_rate = .9
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    past_valid_MSE = 0

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index
            #cost_ij,mse_ij,pnsr_ij = train_model(minibatch_index,lr)
            cost_ij,mse_ij,pnsr_ij = train_model(minibatch_index)
            if (iter % 100 == 0) and verbose:
                print('training @ iter = ' + str(iter) + "\tcost = " + str(cost_ij) + 
                    "\tmse/pixel = " + str(mse_ij) +
                    "\t pnsr = " + str(pnsr_ij))

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = []
                validation_reconstructed = []
                validation_pnsr = []
                validation_MSE = []
                for i in xrange(n_valid_batches):
                    valid_output = validate_model(i)
                    validation_losses.append(valid_output[0])
                    validation_MSE.append(valid_output[1])
                    validation_pnsr.append(valid_output[2])
                    validation_reconstructed.append(valid_output[3])
                
                this_validation_loss = np.mean(validation_losses)

                
                print('epoch %i, minibatch %i/%i, validation cost %f mse/pixel: %f pnsr: %f' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss,
                         np.mean(validation_MSE),
                         np.mean(validation_pnsr)))

                if round(np.mean(validation_MSE)) == past_valid_MSE:
                     new_rate = decay_learning_rate_function()
                     print('new learning rate:')
                     print(new_rate)

                past_valid_MSE = round(np.mean(validation_MSE))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = []
                    test_reconstructed = []
                    test_pnsr = []
                    test_MSE = []
                    for i in xrange(n_test_batches):
                        test_output = test_model(i)
                    
                        test_losses.append(test_output[0])
                        test_MSE.append(test_output[1])
                        test_pnsr.append(test_output[2])
                        test_reconstructed.append(test_output[3])
                    test_score = np.mean(test_losses)

                    f,axiss = plt.subplots(4, 2)
                    i = 0
                    #print (len(test_output))
                    '''
                    while i < 1:
                        #print(len(test_reconstructed))
                        #print('\n')
                        #print(type(test_reconstructed[0][i]))
                        #print(test_reconstructed[0][i].shape)
                        im=np.reshape(test_reconstructed[0][i],(3,output_len,output_len))

                        im = im.transpose(1,2,0)
                        plt.subplot(4,2,i+1)
                        plt.imshow(im)
                        i = i + 1
                    f.savefig('/mnt/output/srcnn_{0}.png'.format(epoch, i))
                    plt.close(f)
                    '''
                    
                    print(('     epoch %i, minibatch %i/%i, test cost of '
                               'best model %f perpixel mse %f and test pnsr %f') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score,
                               np.mean(test_MSE),
                               np.mean(test_pnsr)))


    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation pnsr of %f obtained at iteration %i, '
          'with test cost %f perpixel mse %f test pnsr %f' %
          (best_validation_loss, best_iter + 1, test_score,np.mean(test_MSE),np.mean(test_pnsr)))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
    
    
def train_nn_augmented(train_set_x, train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,output_len,decay_learning_rate_function,
            verbose = False,batch_size = 50, flip_p=0, rotate_p=0, translate_p=0, noise_p=0):
    """
    Wrapper function for training and test THEANO model

    :type train_model: Theano.function
    :param train_model:

    :type validate_model: Theano.function
    :param validate_model:

    :type test_model: Theano.function
    :param test_model:

    :type n_train_batches: int
    :param n_train_batches: number of training batches

    :type n_valid_batches: int
    :param n_valid_batches: number of validation batches

    :type n_test_batches: int
    :param n_test_batches: number of testing batches

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to

    """

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    decay_rate = .9
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    past_valid_MSE = 0

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index
            #cost_ij,mse_ij,pnsr_ij = train_model(minibatch_index,lr)
            temp_data = train_set_x.get_value() 
            data = temp_data[minibatch_index * batch_size:  (minibatch_index+1) *batch_size]
            
            if(flip_p):
                data = flip_image(data)
            if(noise_p == 1):
                data = noise_image(data)
            if(noise_p == 2):
                data = noise_image(data, gaussian_noise = False)
            if(rotate_p):
                data = rotate_image(data)
            if(translate_p):
                data = translate_image(data)

            cost_ij,mse_ij,pnsr_ij = train_model(data, minibatch_index)
            if (iter % 100 == 0) and verbose:
                print('training @ iter = ' + str(iter) + "\tcost = " + str(cost_ij) + 
                    "\tmse/pixel = " + str(mse_ij) +
                    "\t pnsr = " + str(pnsr_ij))
            
            if (iter + 1) % validation_frequency == 0:
                
                

                # compute zero-one loss on validation set
                validation_losses = []
                validation_reconstructed = []
                validation_pnsr = []
                validation_MSE = []
                for i in xrange(n_valid_batches):
                    valid_output = validate_model(i)
                    validation_losses.append(valid_output[0])
                    validation_MSE.append(valid_output[1])
                    validation_pnsr.append(valid_output[2])
                    validation_reconstructed.append(valid_output[3])
                
                this_validation_loss = np.mean(validation_losses)

                
                print('epoch %i, minibatch %i/%i, validation cost %f mse/pixel: %f pnsr: %f' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss,
                         np.mean(validation_MSE),
                         np.mean(validation_pnsr)))

                if round(np.mean(validation_MSE)) == past_valid_MSE:
                     new_rate = decay_learning_rate_function()
                     print('new learning rate:')
                     print(new_rate)

                past_valid_MSE = round(np.mean(validation_MSE))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = []
                    test_reconstructed = []
                    test_pnsr = []
                    test_MSE = []
                    for i in xrange(n_test_batches):
                        test_output = test_model(i)
                    
                        test_losses.append(test_output[0])
                        test_MSE.append(test_output[1])
                        test_pnsr.append(test_output[2])
                        test_reconstructed.append(test_output[3])
                    test_score = np.mean(test_losses)

                    f,axiss = plt.subplots(4, 2)
                    i = 0
                    #print (len(test_output))
                    '''
                    while i < 1:
                        #print(len(test_reconstructed))
                        #print('\n')
                        #print(type(test_reconstructed[0][i]))
                        #print(test_reconstructed[0][i].shape)
                        im=np.reshape(test_reconstructed[0][i],(3,output_len,output_len))

                        im = im.transpose(1,2,0)
                        plt.subplot(4,2,i+1)
                        plt.imshow(im)
                        i = i + 1
                    f.savefig('/mnt/output/srcnn_{0}.png'.format(epoch, i))
                    plt.close(f)
                    '''
                    
                    print(('     epoch %i, minibatch %i/%i, test cost of '
                               'best model %f perpixel mse %f and test pnsr %f') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score,
                               np.mean(test_MSE),
                               np.mean(test_pnsr)))


    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation pnsr of %f obtained at iteration %i, '
          'with test cost %f perpixel mse %f test pnsr %f' %
          (best_validation_loss, best_iter + 1, test_score,np.mean(test_MSE),np.mean(test_pnsr)))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))