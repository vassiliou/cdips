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
#import fcn_input as inpt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '../../fcn_train_log',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 400000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

NUM_CLASSES = 2

#tf.app.flags.DEFINE_string('vgg_path','/Users/gus/CDIPS/uns/fcn_model/vgg16.npy',"""Path to the file containing vgg weights""")


def train():
  """Train model for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get the bottlennecked  data.
    fc6_batch, pool_batch, mask_batch = model.inputs()
    
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits,prediction = model.inference((fc6_batch,pool_batch,mask_batch))
    
    # Calculate loss.
    loss = model.loss(logits, mask_batch,NUM_CLASSES)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.train(loss, global_step)

    #dice score

    def dice(prediction,ground_truth):
      denom =  prediction.sum() + ground_truth.sum()
      if denom == 0:
        return 1
      else:
        cap = np.logical_and(prediction,ground_truth)
        n_cap = cap.sum()
        return 2*n_cap/denom

    def batch_dice(pred_batch, label_batch):
      ### we should probably vectorize this...
      assert pred_batch.shape[0] == label_batch.shape[0]
      pred_batch = pred_batch.reshape([-1,model.PREDICTION_SHAPE[0],model.PREDICTION_SHAPE[1]])
      label_batch = label_batch.reshape([-1,model.PREDICTION_SHAPE[0],model.PREDICTION_SHAPE[1]])
      batch_dice=[]
      for ix in range(label_batch.shape[0]-1):
        batch_dice.append( dice(pred_batch[ix,:,:],label_batch[ix,:,:]) )
      return np.array(batch_dice).mean()

        
    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

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
    
    
    # Start the queue runners.
    print('Starting queue runners...', end='')
    tf.train.start_queue_runners(sess=sess)
    print('done')
    
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      #test = sess.run(fuse_pool)
      #print(test.shape)
      #print(fc6_batch.get_shape())
      #print(pool_batch.get_shape())
      #print(mask_batch.get_shape())
      print('About to run...', end='')
      _, loss_value,predict,ground_truth = sess.run([train_op, loss,prediction,mask_batch])
      print('done')
      #print(sess.run(images)[0])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 25 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)
        batched_dice = batch_dice(predict,ground_truth)
        format_str = ('%s: step %d, batch loss = %.2f, batch dice score = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             batched_dice, examples_per_sec, sec_per_batch))

      if step % 5000 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 40000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)



def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
  tf.app.run()
