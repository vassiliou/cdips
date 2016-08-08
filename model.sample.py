import fcn_input
import fcn_model

#import fcn_eval


modelname = '3457'
# Ex. modelname = '3457'
# Look in training.bin for column model_3457
# Load if found
# otherwise?  error?


#NUM_CLASSES = aws_fcn_input.NUM_CLASSES
NUM_CLASSES = 2
BATCH_SIZE = 15

PREDICTION_SHAPE = [420,580,NUM_CLASSES]
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = aws_fcn_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = aws_fcn_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of records to process in a batch.""")
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

# Starting at 40e-6, ~8000 steps to plateau
INITIAL_LEARNING_RATE = 4*10e-6

checkpoint = None # or setting[folder].


# post processing
DECISION_BOUNDARY = 0.6
PIXEL_CUTOFF = 500
CONVEX_HULL = False
PCA = False
