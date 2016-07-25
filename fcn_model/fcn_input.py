from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_data_files', 3,
                            """Number of datafiles in our data directory.""")





# Global constants describing the  data set.
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 20000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def read_record(filename_queue):
    class BottleneckRecord(object):
        pass
    result = BottleneckRecord()
    result.mask_height = 416
    result.mask_width = 576
    result.mask_depth = 2
    result.fc6_height=13
    result.fc6_width=18
    result.fc6_depth=4096
    result.pool_height=26
    result.pool_width=36
    result.pool_depth=512
    fc6_len = result.fc6_height*result.fc6_width*result.fc6_depth
    pool_len = result.pool_height*result.pool_width*result.pool_depth
    mask_len = result.mask_height*result.mask_width*result.mask_depth
    record_len = fc6_len + pool_len + mask_len

    reader = tf.FixedLengthRecordReader(record_bytes=4*record_len)
    result.key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.float32)
    #print(record_bytes.get_shape())
    result.fc6 = tf.reshape(tf.slice(record_bytes, [0], [fc6_len]),[result.fc6_height, result.fc6_width, result.fc6_depth])
    result.pool = tf.reshape(tf.slice(record_bytes, [fc6_len], [pool_len]),[result.pool_height, result.pool_width, result.pool_depth])
    float_mask = tf.reshape(tf.slice(record_bytes, [pool_len+fc6_len], [mask_len]),[result.mask_height, result.mask_width, result.mask_depth])
    result.mask = tf.cast(float_mask,tf.uint8)
    return result


def _generate_bottlenecked_batch(fc6, pool, mask, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    fc6_batch, pool_batch, mask_batch = tf.train.shuffle_batch(
        [fc6, pool, mask],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    fc6_batch, pool_batch, mask_batch = tf.train.batch(
        [fc6,pool, mask],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)
  # Display the masks in the visualizer.
  tf.image_summary('masks', 255*mask_batch[:,:,:,:1])
  # print(images.get_shape())
  # print(label_batch.get_shape())
  return fc6_batch, pool_batch, mask_batch


def inputs(data_dir, batch_size,num_data_files=FLAGS.num_data_files):
  """Construct input for evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the MNIST data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

   
  filenames = [os.path.join(data_dir, 'fc6pool4mask_batch_%d' % i)
               for i in xrange(1,num_data_files+1)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)



  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_record(filename_queue)

  # Subtract off the mean and divide by the variance of the pixels??
 
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.1
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)

  #print(min_queue_examples)
  print ('Filling queue with %d bottlenecked inputs before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_bottlenecked_batch(read_input.fc6, read_input.pool,read_input.mask,
                                         min_queue_examples, batch_size,
                                         shuffle=True)

