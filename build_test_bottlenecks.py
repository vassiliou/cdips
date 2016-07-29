from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import glob

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

 
from fcn_model import fcn16_vgg

from skimage import io


## Who is running the script?

user='gus'

### Enter the directory containing the raw input images, and a target directory where to write the predictions:  

SOURCE_DATA_DIRECTORY = '/Users/gus/CDIPS/test_debug_source'

BOTTLE_DATA_DIRECTORY = '/Users/gus/CDIPS/test_debug_bottles'

### Enter path to initial VGG16 weights vgg16.npy

vgg_path = '../vgg16.npy'

### Enter the pattern representing file extension of the raw images:

EXTENSION = '*.tif' 

### Build a sorted list of the filepaths in the source directory:

pattern = os.path.join(SOURCE_DATA_DIRECTORY, EXTENSION)

filepaths = sorted(glob.glob(pattern))

### GPU business

log_device_placement = False

### Helper function to trim and stack raw 420 x 580 grayscale images to 416x576 RGB

def trim_and_rgb(img,trim=2):
    """ img is output from feeding io.imread a .tif tile.
        Returns trimmed rgb """
    grayscale=img[trim:-trim,trim:-trim]
    return np.dstack((grayscale,grayscale,grayscale))

### Function to build bottleneck files, and write them to the BOTTLE_DATA_DIRECTORY:

def build_testing_bottlenecks(file_paths,target_directory,vgg_path=vgg_path,user=user,device=log_device_placement):
    
  """load all *.tif files. For each one, compute bottlenecks, append an array of zeros of length 2*416*576, and write to disk
    Raw image file x.tif corresponds to the bottleneck file x.btl """

  with tf.Graph().as_default():

    init = tf.initialize_all_variables()
    
    batch_images = tf.placeholder(tf.float32,[None, 416, 576, 3])

    net = fcn16_vgg.FCN16VGG(vgg16_npy_path=vgg_path)
    
    with tf.name_scope("content_vgg"):
       net.build(batch_images, train=False, num_classes=2, random_init_fc8=True,debug=False)

       
    bottleneck=[net.fc6, net.pool4]

    ## get a tensorflow session running
    
    if user == 'gus':
        sess = tf.Session()

    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement=device
        sess = tf.Session(config=config)

    init=tf.initialize_all_variables()
    sess.run(init)
            
    ## now build and write the files
    
    for path in file_paths:
            img = io.imread(path)
            imgrbg=np.expand_dims(trim_and_rgb(img),axis=0)
            bottles = sess.run(bottleneck, feed_dict={batch_images:imgrbg})
            record=[b.flatten() for b in bottles]
            ### stick a zero mask on the end since we don't know the ground truth
            record.append(np.zeros(2*416*576))
            flat_record = np.concatenate(record)  
            f_name = os.path.split(path)[1]
            f_name = os.path.splitext(f_name)[0]+'.btl'
            write_path = os.path.join(target_directory,f_name)
            flat_record.astype('float32').tofile(write_path)
            print('{file}'.format(file=write_path))

                                    
def main(argv=None):  # pylint: disable=unused-argument
    print('Building .btl files from raw .tif images in directory: ' + str(SOURCE_DATA_DIRECTORY))
    build_testing_bottlenecks(filepaths,BOTTLE_DATA_DIRECTORY)

if __name__ == '__main__':
  tf.app.run()
