import pandas as pd
import uns


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
    return batch.array_rgb(), batch.array_masks()
    

def read_output(path,layer_dims):
    """ Input: layer_dims, length 3 list of dimensions of the output of bottleneck layer
        Returns ndarray of shape (num_batches, layer_dims) """
    data = pd.read_msgpack(path)
    batch_size = data.index.size
    dims = [batch_size] + layer_dims
    return data.as_matrix().reshape(dims)

