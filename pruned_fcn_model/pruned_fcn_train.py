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

import pruned_fcn_model as model
import pruned_fcn_input as inpt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/Users/gus/CDIPS/uns/fcn_train_log',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

NUM_CLASSES = 2

FC7_SHAPE=[4096,512]
FC8_SHAPE=[512,256]

#tf.app.flags.DEFINE_string('vgg_path','/Users/gus/CDIPS/uns/fcn_model/vgg16.npy',"""Path to the file containing vgg weights""")


def train():
  """Train model for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get the bottlennecked  data.
    fc6_batch, pool_batch, mask_batch = model.inputs()
    
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits,fc7,fc8 = model.inference((fc6_batch,pool_batch,mask_batch))

    
    # Calculate loss.
    loss = model.loss(logits, mask_batch,NUM_CLASSES)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.train(loss, global_step)

    #accuracy = model.accuracy(logits,labels)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    print('Initializing all variables...')
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)
    print('Starting queue runners...')
    
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      #test = sess.run(fuse_pool)
      #print(test.shape)
      #print(fc6_batch.get_shape())
      #print(pool_batch.get_shape())
      #print(mask_batch.get_shape())
      _, loss_value,fc7_val, fc8_val = sess.run([train_op, loss,fc7,fc8])
      print('fc7 shape: ' + str(fc7_val.shape))
      print('fc8 shape: ' + str(fc8_val.shape))

      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 1 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, batch loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)



def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
  tf.app.run()
