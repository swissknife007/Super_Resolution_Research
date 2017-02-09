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

from theano.tensor.nnet import conv

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
from theano.tensor.signal import pool

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

#This conv layer reduces output volume by (W - F + 1)
#TODO: Toggle initialization to He init vs. SRCNN's original N(0,.001)
class Conv_Layer_ReLU(object):
    def __init__(self, rng, input, filter_shape, image_shape):
        self.input = input

        # initialize weights with random weights

        ##Xavier-style init
        init = np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]) * np.sqrt(2.0/image_shape[2]**2)
        
        #Init from paper
        init = np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]) * .01

        self.W = theano.shared(np.asarray(init,dtype=theano.config.floatX),borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.ones((filter_shape[0],), dtype=theano.config.floatX) #* .0001
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode='valid'
        )
        self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

#use Relu, zero padding  
class Conv_Layer_VGG(object):
    def __init__(self, input, filter_shape, image_shape, weights):
        self.input = input
  
        ##load pretrained weights!!
        W,b=weights
        self.W=W
        self.b=b
        
        #print(W.shape)
        #print(b.shape)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode='half'
        )
  
        reshaped_b=b.reshape((1,b.shape[0],1,1))
        
        self.output = T.nnet.relu(conv_out + reshaped_b)

        # store parameters of this layer
        self.params = [self.W, self.b]
        
        # keep track of model input
        self.input = input

class Conv_Layer_None(object):
    def __init__(self, rng, input, filter_shape, image_shape):
        self.input = input

        # initialize weights with random weights
        init = np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]) * np.sqrt(2.0/image_shape[2]**2)
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
            border_mode='valid'
        )
        self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

def train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,output_len,decay_learning_rate_function,
            verbose = True, print_imgs = False):
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
            cost_ij,mse_ij,pnsr_ij,reconstucted_imgs = train_model(minibatch_index)
            
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
                     # print('new learning rate:')
                     # print(new_rate)

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

                    # f,axiss = plt.subplots(4, 2)
                    # i = 0
                    # #print (len(test_output))

                    # while i < 1:
                    #     im=np.reshape(test_reconstructed[0][i],(3,output_len,output_len))

                    #     im = im.transpose(1,2,0)
                    #     plt.subplot(4,2,i+1)
                    #     plt.imshow(im)
                    #     i = i + 1
                    # f.savefig('output/output_SRCNN_ADAM_9_1_5/srcnn_{0}.png'.format(epoch, i))
                    # plt.close(f)

                    
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