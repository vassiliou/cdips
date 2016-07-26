from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


#from . import fcn_model as model 
#from . import fcn_input as inpt

import fcn_model as model
import fcn_input as inpt

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('checkpoint_path', '/Users/gus/CDIPS/model.ckpt-40000',
                           """Path to checkpoint file containing learned weights""")

tf.app.flags.DEFINE_string('log_dir', 'Users/gus/CDIPS/fcn_eval_log',"""Directory to write eval logs """)


tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('max_steps', 100,
                            """Number of eval batches to run.""")


NUM_CLASSES = 2

#tf.app.flags.DEFINE_string('vgg_path','/Users/gus/CDIPS/uns/fcn_model/vgg16.npy',"""Path to the file containing vgg weights""")

def eval():
  """Evaluate the model against cross validation dataset."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get the bottlennecked  data.
    fc6_batch, pool_batch, mask_batch = model.inputs(data_dir = model.eval_dir)

    # Ground truth masks
    mask_labels = tf.split(3, 2,mask_batch)[1]
    
  
    # The model's predictions

    logits,prediction = model.eval_inference((fc6_batch,pool_batch,mask_batch))
    
    # Calculate loss.
    loss = model.loss(logits, mask_batch,NUM_CLASSES)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.

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
    
    summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)

    dice_history=[]
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      #test = sess.run(fuse_pool)
      #print(test.shape)
      #print(fc6_batch.get_shape())
      #print(pool_batch.get_shape())
      #print(mask_batch.get_shape())
      print('About to run...', end='')
      loss_value,preds,labels = sess.run([loss,prediction,mask_labels])
      print('done')
      #print(sess.run(images)[0])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 1 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)
        batch_dice = ndice(preds,labels)
        dice_history.append(batch_dice)
        format_str = ('%s: step %d, batch loss = %.2f, batch dice score = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             batch_dice,examples_per_sec, sec_per_batch))

      if step % 5 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if (step + 1) == FLAGS.max_steps:
        dice_array = np.array(dice_history)
        mean_dice= dice_array.mean()
        print('Finished evaluation: mean dice score ' + str(mean_dice) )
        dice_path=FLAGS.log_dir +'/dice_scores'
        np.save(FLAGS.log_dir +'/dice_scores',dice_array)
        print('Saved eval set dice scores to '+ dice_path)  



def main(argv=None):  # pylint: disable=unused-argument
    eval()


if __name__ == '__main__':
  tf.app.run()
