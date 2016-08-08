

#uns.py


datafolder = "/Users/gus/CDIPS/nerve-project/"
trainbin = "/Users/gus/CDIPS/uns/training.bin"
bottlefolder = '/home/chrisv/code/bottleneck_files'


trainfolder = os.path.join(datafolder, 'train')
testfolder = os.path.join(datafolder, 'test')

# aws_fcn_model

tf.app.flags.DEFINE_string('data_dir', '/home/ubuntu/train_records',

#fcn_train
tf.app.flags.DEFINE_string('train_dir', '/home/ubuntu/train_log',
checkpath = '/home/ubuntu/train_log/model.ckpt-7000

#fcn_eval
RECORD_DIRECTORY = '/home/ubuntu/validation_records'
PREDICTION_DIRECTORY = '/home/ubuntu/validation_predictions'
checkpath = '/home/ubuntu/train_log/model.ckpt-15000'

#RLE_predictions
output_pattern='/home/ubuntu/test_output/*.npy'

#record producer
INPUT_DIRECTORY = '/home/ubuntu/validation'
RECORD_DIRECTORY = '/home/ubuntu/validation_records'

#FCN16VGG
vgg_path = '/home/ubuntu/vgg16.npy'
