import fcn16_vgg 
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import utils
import scipy as scp
import scipy.misc
import loss
import pandas as pd
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt


import uns
#from uns import training


## Start a Tensorflow session

#sess = tf.Session()

# create placeholder for batch images 

#batch_images = tf.placeholder(tf.float32,[None,420,580,3])

# variable to keep track of path to the pre-trained weights

#vgg_path = '../tensorflow-fcn/vgg16.npy'

# build an FCN16 

#net = fcn16_vgg.FCN16VGG(vgg16_npy_path=vgg_path)

#with tf.name_scope("content_vgg"):
       # net.build(batch_images, train=True, num_classes=2, random_init_fc8=True)


# bottlenecked output

#bottleneck_output = net.fc6

def get_bottleneck_dims(bottleneck):
    """ input bottleneck is a tf Variable"""
    return [int(d) for  d in bottleneck.get_shape().dims[1:]]

# code for handling the images


def make_batches(trainingData,batch_size):

    num_images = trainingData.index.size
    batches=[]
    n_batches = num_images//batch_size
    for i in range(n_batches):
        batches.append(uns.batch(trainingData.iloc[range(batch_size*i,batch_size*(i+1))]))
    #batches.append(uns.batch(trainingData.iloc[range(n_batches*batch_size,num_images)]))
    return batches

def load_batch(batch):
    """ Returns an array of images """
    return batch.array_rgb()
    

def read_output(path,layer_dims):
    """ Input: layer_dims, length 3 list of dimensions of the output of bottleneck layer
        Returns ndarray of shape (num_batches, layer_dims) """
    data = pd.read_msgpack(path)
    batch_size = df.index.size
    dims = [batch_size] + layer_dims
    return data.as_matrix().reshape(dims)

