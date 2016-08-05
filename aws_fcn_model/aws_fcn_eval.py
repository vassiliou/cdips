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

import aws_fcn_model as model
import aws_fcn_input as inpt

import skimage.io

FLAGS = tf.app.flags.FLAGS



on_mac = True


### Enter the directory containing the raw input images, and a target directory where to write the predictions:  

    
RECORD_DIRECTORY = '/home/ubuntu/validation_records'
PREDICTION_DIRECTORY = '/home/ubuntu/validation_predictions'

if on_mac:
    RECORD_DIRECTORY = '/Users/gus/Desktop/aws/validation_debug_records'
    PREDICTION_DIRECTORY = '/Users/gus/Desktop/aws/prediction_debug'

### Choose post-processing hyperparameters:

DECISION_BOUNDARY = 0.6

PIXEL_CUTOFF = 500

CONVEX_HULL = False

PCA = False

### Setup paths to model checkpoint directory:

checkpath = '/home/ubuntu/train_log/model.ckpt-15000'

if on_mac:
    checkpath = '/Users/gus/Dropbox/saved_models/aws_first_model.ckpt'


tf.app.flags.DEFINE_string('checkpoint_path', checkpath,
                           """Path to checkpoint file containing learned weights""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

### Enter the pattern representing file extension of the bottleneck files

EXTENSION = '*.rec' 

### Shit you probably don't want to mess with:

NUM_CLASSES = 2

BATCH_SIZE = 1

### Build a sorted list of the filepaths in the source directory:

pattern = os.path.join(RECORD_DIRECTORY, EXTENSION)

filepaths = sorted(glob.glob(pattern))

### Function for post-processing of predicted masks

def post_process(probabilities, d_boundary=DECISION_BOUNDARY, pix_cutoff=PIXEL_CUTOFF, convex_hull=CONVEX_HULL, pca=PCA):

  final_prediction=probabilities

  return final_prediction


def RLE(mask):
  pass

def eval(file_paths,source_dir=RECORD_DIRECTORY,target_directory=PREDICTION_DIRECTORY):
  """Evaluate the model against a test dataset

    Saves predictions and masks as an array of shape [# examples] , 2, 416, 576
    Slice [:,0,:,:] corresponds to predictions and  slice [:,1,:,0] corresponds to ground truth"""


  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get the bottled data.
    
    image_batch, mask_batch = model.inputs(train=False,inpt_dir=source_dir)
    
    # The model's predictions

    logits, prediction = model.inference(image_batch,train=False,random_fc8=False)

    pixel_probabilities = tf.reshape( tf.nn.softmax(tf.reshape(logits, (-1, NUM_CLASSES))) , (-1,210,290,NUM_CLASSES))
    
    # Calculate loss.
    #loss = model.loss(logits, mask_batch,NUM_CLASSES)

    #accuracy = model.accuracy(logits,labels)

    #train_op = model.train(loss, global_step)

    # Build the summary operation based on the TF collection of Summaries.

    summary_op = tf.merge_all_summaries()

    saver = tf.train.Saver(tf.all_variables())

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement=FLAGS.log_device_placement
    sess = tf.Session(config=config)
 
      
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
      probabilities, ground_truth = sess.run([pixel_probabilities,mask_batch])
      probabilities, ground_truth = probabilities[0,:,:,0], ground_truth[0,:,:,0]

      to_write = np.array([probabilities, ground_truth])
      ## Write the prediction to disk
      f_name = os.path.split(file_path)[1]
      f_name = os.path.splitext(f_name)[0]+''
      write_path = os.path.join(target_directory,f_name)
      np.save(write_path,to_write)
      if counter % 10 ==0:
        print('Saving prediction {d} to: '.format(d=counter) + str(write_path) )



def main(argv=None):  # pylint: disable=unused-argument
    print('Running predicion on .rec files in ' + str(RECORD_DIRECTORY))
    eval(filepaths)

if __name__ == '__main__':
  tf.app.run()
