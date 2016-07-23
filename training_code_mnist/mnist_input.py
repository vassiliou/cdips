from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

IMAGE_SIZE = 28


# Global constants describing the MNIST data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 20000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def read_mnist(filename_queue):
    class MNISTRecord(object):
        pass
    result = MNISTRecord()
  
    label_bytes = 10  
    result.height = 28
    result.width = 28
    result.depth = 1
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes
    #print(record_bytes)

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    raw_image = tf.reshape(tf.slice(record_bytes, [0], [image_bytes]),[result.height, result.width, result.depth])
    result.image = tf.cast(raw_image,tf.float32)/255
    result.label = tf.slice(record_bytes, [image_bytes], [label_bytes])
    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size,shuffle):
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
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)
  # Display the training images in the visualizer.
  tf.image_summary('images', images)
  # print(images.get_shape())
  # print(label_batch.get_shape())
  return images, tf.reshape(label_batch, [batch_size,10])


def inputs(data_dir, batch_size):
  """Construct input for evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the MNIST data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [os.path.join(data_dir, 'data_chunk_%d.bin' % i)
               for i in xrange(1, 6)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)



  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_mnist(filename_queue)
  image = tf.cast(read_input.image, tf.float32)

  # Subtract off the mean and divide by the variance of the pixels??
 
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.1
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  #print(min_queue_examples)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)

