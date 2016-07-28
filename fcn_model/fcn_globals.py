# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 21:20:50 2016

@author: chrisv
"""

### FROM fcn_eval
import os

checkpath = '/Users/gus/CDIPS/model.ckpt-40000'
outpath = '/Users/gus/CDIPS/fcn_eval_log'

if os.environ['USER'] == 'chrisv':
    print(os.environ['USER'], end='')
    if os.environ['SESSION'] == 'Lubuntu':
        print(" on Lubuntu")        
        checkpath = '/home/chrisv/code/fcn_train_log/model.ckpt-10000'
        outpath = '/home/chrisv/code/fcn_train_log/'
    else:
        print(" on Mac")

NUM_CLASSES = 2

#
# fcn_input
#

user = 'gus'

if user == 'gus':
    bottle_files = '/Users/gus/CDIPS/mask_debug/'
else:
    bottle_files = '/home/chrisv/code/bottleneck_files'
    
#tf.app.flags.DEFINE_string('train_data_dir', bottle_files,
#                           """Path to the training input data directory.""")    
#
#tf.app.flags.DEFINE_string('eval_dir', bottle_files,
#                           """Path to the training input data directory.""")    
#    
#
#    tf.app.flags.DEFINE_integer('num_train_files',len(train_idx) ,
#                          """Number of training files in our data directory.""")
#    tf.app.flags.DEFINE_integer('num_eval_files',len(validate_idx) ,
#                          """Number of crossvalidation files in our data directory.""")

#
# fcn_train
#

#FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('train_log_dir', '../../fcn_train_log',
#                           """Directory where to write event logs """
#                           """and checkpoint.""")
#
#tf.app.flags.DEFINE_integer('batch_size', 10,
#                            """Number of records to process in a batch.""")
#
#tf.app.flags.DEFINE_integer('max_steps', 100000,
#                            """Number of batches to run.""")
#tf.app.flags.DEFINE_boolean('log_device_placement', False,
#                            """Whether to log device placement.""")

NUM_CLASSES = 2

#
# fcn_model
#

#NUM_CLASSES = fcn_input.NUM_CLASSES
PREDICTION_SHAPE = [416,576,NUM_CLASSES]
#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = fcn_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = fcn_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 10e-6       # Initial learning rate.




