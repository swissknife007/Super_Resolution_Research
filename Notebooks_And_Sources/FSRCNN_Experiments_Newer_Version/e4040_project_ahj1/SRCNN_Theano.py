import sys
import numpy as np
from theano.tensor.nnet import conv
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
from theano.tensor.signal import pool
import theano
import theano.tensor as T
import os
import scipy.ndimage


from SRCNN_Layers import *

def train_SRCNN(train_set_x,train_set_y,valid_set_x,valid_set_y,test_set_x,test_set_y,
    n_train_batches, n_valid_batches, n_test_batches, n_epochs, batch_size,learning_rate,upsampling_factor=4):
    #Assume x to be shape (batch_size,3,33,33)
    x = T.matrix('x')
    y = T.matrix('y')

    theano.config.optimizer = 'fast_compile'
    print "theano optimizer: " + str(theano.config.optimizer)

    rng = np.random.RandomState(11111)
    index = T.lscalar() 

    reshaped_input = x.reshape((batch_size,3,33,33))
    reshaped_gt = y.reshape((batch_size,3,33,33))

    #Upsampling layer now done in preprocessing to save compute
    #upsampled_input = T.nnet.abstract_conv.bilinear_upsampling(reshaped_input,upsampling_factor,batch_size=batch_size,num_input_channels=3)
    # r_fun = theano.function([index],upsampled_input.shape,givens = {
    #         x: train_set_x[index * batch_size: (index + 1) * batch_size]
    #         })
    # theano.printing.debugprint(r_fun(0))
    
    #Filter params
    f1 = 9
    f2 = 1
    f3 = 5
    output_len = 33 - f1 - f2 - f3 + 3
    #Conv for Patch extraction
    conv1 = Conv_Layer_ReLU(rng, reshaped_input, image_shape=(batch_size,3,33,33),filter_shape = (64,3,f1,f1))
    conv1_len = 33 - f1 + 1 
    #Conv for Non linear mapping

    conv2 = Conv_Layer_ReLU(rng, conv1.output, image_shape=(batch_size,64,conv1_len,conv1_len),filter_shape = (32,64,f2,f2))
    conv2_len = conv1_len - f2 + 1
    #Conv for Reconstruction
    conv3 = Conv_Layer_None(rng, conv2.output, image_shape=(batch_size,32,conv2_len,conv2_len),filter_shape = (3,32,f3,f3))
    model_output = conv3.output

    sub_y = reshaped_gt[:,:,:output_len,:output_len]

    #MSE between center pixels of prediction and ground truth
    cost = 1.0/batch_size * T.sum((sub_y-model_output) ** 2)

    #Perchannel cost 
    # costs = []
    # for d in sub_y.shape[0]:
    #     channel_cost = cost = 1.0/batch_size * T.sum((sub_y[d,:,:]-model_output[d,:,:]) ** 2)
    #     costs.append(channel_cost)

    params = conv3.params + conv2.params + conv1.params

    #ADAM opt
    beta1 =theano.shared(np.cast[theano.config.floatX](0.9), name='beta1')
    beta2 =theano.shared(np.cast[theano.config.floatX](0.999), name='beta2')
    eps =theano.shared(np.cast[theano.config.floatX](1e-8), name='eps')
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value()*np.cast[theano.config.floatX](0.))
        m = theano.shared(param.get_value()*np.cast[theano.config.floatX](0.))    
        v = theano.shared(param.get_value()*np.cast[theano.config.floatX](0.))    

        updates.append((param, param - learning_rate*param_update))
        updates.append((param_update, m/(T.sqrt(v) + eps)))
        updates.append((m, beta1 * m + (np.cast[theano.config.floatX](1.) - beta1) * T.grad(cost, param)))
        updates.append((v, beta2 * v + (np.cast[theano.config.floatX](1.) - beta2) * T.sqr(T.grad(cost, param))))

    #PSNR of a patch is based on color space
    MSE_per_pixel = cost/(output_len*output_len*3) 
    psnr = 20 * T.log10(255) - 10 * T.log10(MSE_per_pixel)
    reconstucted_imgs = model_output
    
    #Theano function complilation
    #if neccessary, could load here
    test_model = theano.function(
        [index],
        [cost,MSE_per_pixel,psnr,reconstucted_imgs],
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        [cost,MSE_per_pixel,psnr,reconstucted_imgs],
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_model = theano.function(
        [index],
        [cost,MSE_per_pixel,psnr],
        updates=updates,
        givens={
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        })

    train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,output_len,
            verbose = True)

    return validate_model,test_model

def load_dataset(dirname,data_type = 'data_x'):
    
    up_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(up_dir,dirname)
    print data_dir

    if not os.path.isfile(data_dir+'.npz'):
        print "creating npz file"
        dataset_size = len(os.listdir(data_dir))
        if data_type == 'data_x':
            data = np.zeros((dataset_size,3,8,8))
        else:
            data = np.zeros((dataset_size,3,33,33))
        for root, dirs, files in os.walk(data_dir, topdown=False):
            for counter,name in enumerate(files):
                full_filename = os.path.join(root, name)
                img = scipy.ndimage.imread(full_filename) #(8,8,3)
                img = np.transpose(img,(2,0,1))
                data[counter,:,:,:] = img

        np.savez(data_dir + '.npz',data=data)
    else:
        print "loading from npz"
        data = np.load(data_dir + '.npz')['data']
    return data


###Debugging function
    # r_fun = theano.function([index],[index,reconstucted_imgs,cost],givens = {
    #         x: train_set_x[index * batch_size: (index + 1) * batch_size],
    #         y: train_set_y[index * batch_size: (index + 1) * batch_size]
    #         })
    # theano.printing.debugprint(r_fun(0))