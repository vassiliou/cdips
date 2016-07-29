from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import glob

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

#from . import fcn_model as model 
#from . import fcn_input as inpt

import fcn_model as model
import fcn_input as inpt

import skimage.io

FLAGS = tf.app.flags.FLAGS


### Who's running the script?

user = 'notgus'

### Enter the directory containing the raw input images, and a target directory where to write the predictions:  
if user == 'gus':
    
    BOTTLE_DATA_DIRECTORY = '/Users/gus/CDIPS/test_debug_bottles'
    TARGET_DATA_DIRECTORY = '/Users/gus/CDIPS/test_debug_output'

else:
    BOTTLE_DATA_DIRECTORY = '/home/chrisv/code/train_bottles'
    TARGET_DATA_DIRECTORY = '/home/chrisv/code/train_output'


### Choose post-processing hyperparameters:

DECISION_BOUNDARY = 0.6

PIXEL_CUTOFF = 500

CONVEX_HULL = False

PCA = False

### Setup paths to model checkpoint directory:

checkpath = '/Users/gus/CDIPS/model.ckpt-40000'

if os.environ['USER'] == 'chrisv':
    print(os.environ['USER'], end='')
    if os.environ['SESSION'] == 'Lubuntu':
        print(" on Lubuntu")        
        checkpath = '/home/chrisv/code/fcn_train_log/BatchOf10/model.ckpt-10000'
        outpath = '/home/chrisv/code/fcn_train_log/'
    else:
        print(" on Mac")

tf.app.flags.DEFINE_string('checkpoint_path', checkpath,
                           """Path to checkpoint file containing learned weights""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

### Enter the pattern representing file extension of the bottleneck files

EXTENSION = '*.btl' 

### Shit you probably don't want to mess with:

NUM_CLASSES = 2

tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of records to process in a batch. This should
always be 1 in this script in order to ensure we only write one prediction per file.""")


### Build a sorted list of the filepaths in the source directory:

pattern = os.path.join(BOTTLE_DATA_DIRECTORY, EXTENSION)

filepaths = sorted(glob.glob(pattern))

### Function for post-processing of predicted masks

def post_process(probabilities, d_boundary=DECISION_BOUNDARY, pix_cutoff=PIXEL_CUTOFF, convex_hull=CONVEX_HULL, pca=PCA):

  final_prediction=probabilities

  return final_prediction


def RLE(mask):
  pass

def eval(file_paths,target_directory=TARGET_DATA_DIRECTORY):
  """Evaluate the model against a test dataset

    Saves predictions and masks as an array of shape [# examples] , 2, 416, 576
    Slice [:,0,:,:] corresponds to predictions and  slice [:,1,:,0] corresponds to ground truth"""


  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get the bottled data.
    
    fc6_batch, pool_batch, mask_batch = model.inputs(data_dir=None, train=False,fnames=file_paths)
    
    # The model's predictions

    logits, prediction = model.inference((fc6_batch,pool_batch,mask_batch),train=False)

    pixel_probabilities = tf.reshape( tf.nn.softmax(tf.reshape(logits, (-1, NUM_CLASSES))) , (-1,416,576,NUM_CLASSES))
    
    # Calculate loss.
    #loss = model.loss(logits, mask_batch,NUM_CLASSES)

    #accuracy = model.accuracy(logits,labels)

    #train_op = model.train(loss, global_step)

    # Build the summary operation based on the TF collection of Summaries.

    summary_op = tf.merge_all_summaries()

    saver = tf.train.Saver(tf.all_variables())

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    if user !='gus':   
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.log_device_placement=FLAGS.log_device_placement
      sess = tf.Session(config=config)
    else:
      sess=tf.Session()
      
    # Start running operations on the Graph.
    print('Initializing all variables...', end='')
    sess.run(init)
    print('done')

    # Restore the model from the checkpoint

    saver.restore(sess,FLAGS.checkpoint_path)
    
    # Start the queue runners.
    print('Starting queue runners...', end='')
    tf.train.start_queue_runners(sess=sess)
    print('done')
    
    ### The main loop: for each filename, 
    
    for counter, file_path in enumerate(file_paths):    
      probabilities= sess.run(pixel_probabilities)[0,:,:,0]
      #print(probabilities.shape)

      final_prediction = post_process(probabilities)

      ## Write the prediction to disk
      f_name = os.path.split(file_path)[1]
      f_name = os.path.splitext(f_name)[0]+'_prediction'
      write_path = os.path.join(target_directory,f_name)
      np.save(write_path,final_prediction)
      if counter % 10 ==0:
        print('Saving prediction {d} to: '.format(d=counter) + str(write_path) )



def main(argv=None):  # pylint: disable=unused-argument
    print('Running predicion on .btl files in ' + str(BOTTLE_DATA_DIRECTORY))
    eval(filepaths)

if __name__ == '__main__':
  tf.app.run()
