from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import re
import sys
import mnist_input


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/Users/gus/CDIPS/practice-training/',
                           """Path to the CIFAR-10 data directory.""")
# Global constants describing the MNIST data set.
IMAGE_SIZE = mnist_input.IMAGE_SIZE
NUM_CLASSES = mnist_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = mnist_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = mnist_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 10e-4       # Initial learning rate.

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

  
def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = tf.Variable(tf.truncated_normal_initializer(shape,stddev=stddev),name=name)
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var




def inputs():
    if not FLAGS.data_dir:
          raise ValueError('Please supply a data_dir')
    data_dir = FLAGS.data_dir
    return mnist_input.inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)


### helpers to build layers

def weight_variable(shape,name):
    initial= tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)

def convolve(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def pool_layer(x,name):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')



def inference(image_batch):
  """Build our MNIST model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    W_conv1 = weight_variable([3,3,1,32],name='weights')
    b_conv1 = bias_variable([32],name='bias')
    h_conv1 = tf.nn.relu(convolve(image_batch,W_conv1) + b_conv1, name='activation')
    _activation_summary(h_conv1)
    pool1 = pool_layer(h_conv1,name='pool1')

  with tf.variable_scope('conv2') as scope:
    W_conv2 = weight_variable([3,3,32,64],name='weights')
    b_conv2 = bias_variable([64],name='bias')
    h_conv2 = tf.nn.relu(convolve(pool1, W_conv2)+b_conv2,name='activation')
    _activation_summary(h_conv2)
    pool2   = pool_layer(h_conv2,name='pool2')

  with tf.variable_scope('conv3') as scope:
    W_conv3 = weight_variable([3,3,64,64],name='weights')
    b_conv3 = bias_variable([64],name='bias')
    h_conv3 = tf.nn.relu(convolve(pool2,W_conv3)+b_conv3,name='activation')
    pool3 = pool_layer(h_conv3,name='pool3')
  
  with tf.variable_scope('fc4') as scope:
    fc1_input = tf.reshape(pool3,[-1,16*64])
    W_fc1 = weight_variable(shape=[16*64,1024],name = 'weights')
    b_fc1 = bias_variable(shape=[1024],name = 'bias')
    h_fc1 = tf.nn.relu(tf.matmul(fc1_input,W_fc1)+b_fc1,name='activations')
    _activation_summary(h_fc1)

  with tf.variable_scope('output') as scope:
    W_fc2 = weight_variable(shape=[1024,10],name='weights')
    b_fc2 = bias_variable(shape=[10],name = 'bias')
    logits = tf.add(tf.matmul(h_fc1,W_fc2),b_fc2,name='logits')
    _activation_summary(logits)
 
  return logits


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.float32)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def accuracy(logits,labels):
  pred = tf.nn.softmax(logits)
  correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(labels,1))
  acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
  return acc



def _add_loss_summaries(total_loss):
  """Add summaries for losses in model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train the model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
