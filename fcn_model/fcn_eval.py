from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path


import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


#from . import fcn_model as model 
#from . import fcn_input as inpt

import fcn_model as model
import fcn_input as inpt

FLAGS = tf.app.flags.FLAGS


checkpath = '/Users/gus/CDIPS/model.ckpt-40000'
outpath = '/Users/gus/CDIPS/fcn_eval_log'

if os.environ['USER'] == 'chrisv':
    print(os.environ['USER'], end='')
    if os.environ['SESSION'] == 'Lubuntu':
        print(" on Lubuntu")        
        checkpath = '/home/chrisv/code/fcn_train_log/model.ckpt-39999'
        outpath = '/home/chrisv/code/fcn_train_log/'
    else:
        print(" on Mac")

tf.app.flags.DEFINE_string('checkpoint_path', checkpath,
                           """Path to checkpoint file containing learned weights""")

tf.app.flags.DEFINE_string('out_dir', outpath,"""Directory to write eval output """)


tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('max_steps', 16,
                            """Number of eval batches to run.""")


NUM_CLASSES = 2

#tf.app.flags.DEFINE_string('vgg_path','/Users/gus/CDIPS/uns/fcn_model/vgg16.npy',"""Path to the file containing vgg weights""")

def eval():
  """Evaluate the model against cross validation dataset.

    Saves predictions and masks as an array of shape [# examples] , 2, 416, 576
    Slice [:,0,:,:] corresponds to predictions and  slice [:,1,:,:] corresponds to ground truth"""


  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get the bottlennecked  data.
    fc6_batch, pool_batch, mask_batch = model.inputs()

    # Ground truth masks
    mask_labels = tf.split(3, 2,mask_batch)[1]
    
  
    # The model's predictions

    logits,prediction = model.eval_inference((fc6_batch,pool_batch,mask_batch))
    
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

    
    # Start running operations on the Graph.
    sess = tf.Session(config=config)
    print('Initializing all variables...', end='')
    sess.run(init)
    print('done')

    # Restore the model from the checkpoint

    saver.restore(sess,FLAGS.checkpoint_path)


    # Dice scores for the eval data
    def ndice(pred_batch,label_batch):
      n_batch = pred_batch.shape[0]
      preds = pred_batch.reshape([n_batch,model.PREDICTION_SHAPE[0],model.PREDICTION_SHAPE[1]])
      labels = label_batch.reshape([n_batch,model.PREDICTION_SHAPE[0],model.PREDICTION_SHAPE[1]])
      denoms = np.sum(preds,axis=(1,2)) + np.sum(labels,axis=(1,2))
      cap = np.logical_and(preds, labels)
      numerators = 2*np.sum(cap, axis=(1,2))
      zero_denoms = np.where(denoms ==0)
      nonzero_denoms = np.where (denoms !=0)
      result = np.empty(denoms.shape)
      result[zero_denoms] = 1
      result[nonzero_denoms] = np.divide(numerators[nonzero_denoms],denoms[nonzero_denoms])
      return result.mean()

    
    # Start the queue runners.
    print('Starting queue runners...', end='')
    tf.train.start_queue_runners(sess=sess)
    print('done')
    
    summary_writer = tf.train.SummaryWriter(FLAGS.out_dir, sess.graph)


    output_records = []
    
    for step in xrange(FLAGS.max_steps):

      preds,labels = sess.run([prediction,mask_labels])
      labels = labels.reshape(labels.shape[:3])
      #print(labels.shape)
      output_records.append(np.array([preds,labels]))

      if (step + 1) == FLAGS.max_steps:
        print('Finished evaluation.')
        out_path=FLAGS.out_dir +'/eval_output'
        collection = np.array(output_records)
        shaped_collection = collection.reshape([-1,2,416,576])
        np.save(out_path,shaped_collection)
        print('Saved cross validation set masks and predictions to: ', out_path)  



def main(argv=None):  # pylint: disable=unused-argument
    eval()


if __name__ == '__main__':
  tf.app.run()
