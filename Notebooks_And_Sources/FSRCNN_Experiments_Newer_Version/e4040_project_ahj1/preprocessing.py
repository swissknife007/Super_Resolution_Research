#This file preprocesses the 91 images used in SRCNN for training 
#1. 91 images are split into subimages of size 33x33 at a stride of 14 -> 24.8k subimages
#2. Each subimage is blurred with a Gaussian (following Johnson we use sigma = 1) 
#3. Each subimage is then subsampled by a given upsampling factor (3,4,6) using bicubic interpolation
## During training, the model will then upsample

import numpy as np
from scipy import ndimage,misc

import os 
import sys

#output_type can also be 'RGB'
def create_subimages(image_folder, output_folder, output_type = 'YCbCr',upsampling_factor = 4):
	for (dirpath,dirnames,filenames) in os.walk(image_folder):
		print image_folder
		for counter,image_filename in enumerate(filenames):
			if image_filename.split('.')[-1] == 'bmp' and image_filename[0] != '.':
				if counter % 10 == 0:
					print "processed:" + str(counter)

				image = misc.imread(os.path.join(image_folder,image_filename),flatten=False, mode = output_type)
				
				#(width,height,channel_depth)
				for w in xrange(int((image.shape[0]-33)/14) + 1):
					for h in xrange(int((image.shape[1]-33)/14) + 1):
						subimage = image[14*w:14*w+33,14*h:14*h+33,:]

						blurred_sub = subimage
						for d in xrange(image.shape[2]):
							blurred_sub[:,:,d] = ndimage.gaussian_filter(subimage[:,:,d],sigma = 1)

						subsampled_subimage = misc.imresize(blurred_sub,1.0/upsampling_factor,interp = 'bicubic')
						misc.imsave(os.path.join(output_folder+"_gt",image_filename.split('.')[0] +"_" + str(w) +"_" +str(h) + '.bmp'),subimage)
						misc.imsave(os.path.join(output_folder,image_filename.split('.')[0] +"_" + str(w) +"_" +str(h) + '.bmp'),subsampled_subimage)

def upsample(data_x):
	upsampled_x = np.zeros((data_x.shape[0],3,33,33))
	for i in xrange(data_x.shape[0]):
	    img = np.transpose(data_x[i,:,:,:],(1,2,0))
	    upsampled_x[i,:,:,:] = np.transpose(misc.imresize(img,(33,33,3),interp = 'bicubic'),(2,0,1))
	return upsampled_x

if __name__ == '__main__':
	#mode can be 'YCbCr', 'RGB'
	create_subimages('./Training_Full',
		'./Data/Training_Subsamples_RGB_4',
		output_type ='RGB',upsampling_factor = 4)
