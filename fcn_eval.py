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


### Constants defining how we run our evaluations

tf.app.flags.DEFINE_integer('batch_size', 10,
                            """Number of records to process in a batch.""")

tf.app.flags.DEFINE_integer('max_steps', 50,
                            """Number of eval batches to run.""")
NUM_CLASSES = 2

###

### setup paths to model checkpoint and output directories

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

tf.app.flags.DEFINE_string('checkpoint_path', checkpath,
                           """Path to checkpoint file containing learned weights""")

tf.app.flags.DEFINE_string('out_dir', outpath,"""Directory to write eval output """)


tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")



#tf.app.flags.DEFINE_string('vgg_path','/Users/gus/CDIPS/uns/fcn_model/vgg16.npy',"""Path to the file containing vgg weights""")


def eval(data_dir):
  """Evaluate the model against cross validation dataset.

    Saves predictions and masks as an array of shape [# examples] , 2, 416, 576
    Slice [:,0,:,:] corresponds to predictions and  slice [:,1,:,0] corresponds to ground truth"""


  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get the bottlennecked  data.
    fc6_batch, pool_batch, mask_batch = model.inputs(data_dir=FLAGS.eval_dir, train=False)

    # Ground truth masks
    mask_labels = tf.split(3, 2,mask_batch)[1]
    
  
    # The model's predictions

    logits,prediction = model.inference((fc6_batch,pool_batch,mask_batch),train=False)

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


    
    # Start the queue runners.
    print('Starting queue runners...', end='')
    tf.train.start_queue_runners(sess=sess)
    print('done')
    
    summary_writer = tf.train.SummaryWriter(FLAGS.out_dir, sess.graph)


    output_records = []
    
    for step in xrange(FLAGS.max_steps):    
      probabilities,labels = sess.run([pixel_probabilities,mask_labels])
      labels = labels.reshape(labels.shape[:3])
      print(labels.shape)
      print(probabilities[:,:,:,0].shape)
      to_add = np.array([probabilities[:,:,:,0],labels])
      print( to_add.shape)
      output_records.append(to_add)


      if (step+1) % 4 == 0:
        print('Step {} of evaluation.'.format(step))
        collection = np.array(output_records)
        #print(collection.shape)
        #shaped_collection = collection.reshape([-1,416,576,2])
        savepath = os.path.join(outpath, 'predictions_chunk_{}'.format((step +1)//4))

        ### from the git merge:
        ##np.save(savepath,~shaped_collection.astype(bool))

        np.save(savepath,collection)
        output_records=[]
        print('Saved cross validation set masks and predictions to: ', savepath)  

      
      if (step + 1) == FLAGS.max_steps:
        print('Finished evaluation.')
        collection = np.array(output_records)
        #shaped_collection = collection.reshape([-1,416,576,2])
        savepath = os.path.join(outpath, 'predictions_chunk_0')

        np.save(savepath,collection)
        print('Saved cross validation set masks and predictions to: ', savepath)  



def main(argv=None):  # pylint: disable=unused-argument
    eval(FLAGS.eval_dir)


if __name__ == '__main__':
  tf.app.run()
