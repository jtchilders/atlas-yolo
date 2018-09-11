#!/usr/bin/env python
print('top o the script')
import os,argparse,logging,json,glob,datetime
import yolo_model
from BatchGenerator import BatchGenerator
import loss_func
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras import backend as keras_backend
import tensorflow as tf
from callbacks import TB
import numpy as np
logger = logging.getLogger(__name__)
print('import complete')
print('keras from: %s' % optimizers.__file__)
print('tensorflow from: %s' % tf.__file__)


def create_config_proto(params):
   '''EJ: TF config setup'''
   config = tf.ConfigProto()
   config.intra_op_parallelism_threads = params.num_intra
   config.inter_op_parallelism_threads = params.num_inter
   config.allow_soft_placement         = True
   os.environ['KMP_BLOCKTIME'] = str(params.kmp_blocktime)
   os.environ['KMP_AFFINITY'] = params.kmp_affinity
   return config


def main():
   ''' simple starter program that can be copied for use when starting a new script. '''
   print('start main')

   parser = argparse.ArgumentParser(description='Atlas Training')
   parser.add_argument('--config_file', '-c',
                       help='configuration in standard json format.')
   parser.add_argument('--horovod', default=False,
                       help='use Horovod')
   parser.add_argument('--num_files','-n', default=-1, type=int,
                       help='limit the number of files to process. default is all')
   parser.add_argument('--lr', default=0.01, type=int,
                       help='learning rate')
   parser.add_argument('--num_intra', type=int,
                       help='num_intra')
   parser.add_argument('--num_inter', type=int,
                       help='num_inter')
   parser.add_argument('--kmp_blocktime', type=int, default=10,
                       help='KMP BLOCKTIME')
   parser.add_argument('--kmp_affinity', default='granularity=fine,verbose,compact,1,0',
                       help='KMP AFFINITY')
   args = parser.parse_args()

   if args.horovod:
      print("importing hvd")
      import horovod.keras as hvd
      print('horovod from: %s' % hvd.__file__)
      print('hvd init')
      hvd.init()
      bcast_global_variables_op = hvd.broadcast_global_variables(0)
      print("Rank:",hvd.rank())
      logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s:' + '{:05d}'.format(hvd.rank()) + ':%(name)s:%(thread)s:%(message)s')
   else:
      logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s:%(name)s:%(thread)s:%(message)s')


   # load configuration
   config_file = json.load(open(args.config_file))

   # get file list
   filelist = glob.glob(config_file['data_handling']['input_file_glob'])
   logger.info('found %s input files',len(filelist))
   if len(filelist) < 2:
      raise Exception('length of file list needs to be at least 2 to have train & validate samples')

   nfiles = len(filelist)
   if args.num_files > 0:
      nfiles = args.num_files

   train_file_index = int(config_file['data_handling']['training_to_validation_ratio'] * nfiles)
   np.random.shuffle(filelist)

   model = yolo_model.build_model(config_file)
   model.summary()

   logger.info('grid (w,h) = (%s,%s)',config_file['training']['gridW'],config_file['training']['gridH'])
   loss_func.set_config(config_file)

   train_imgs = filelist[:train_file_index]
   train_gen = BatchGenerator(config_file,train_imgs)
   valid_imgs = filelist[train_file_index:nfiles]
   valid_gen = BatchGenerator(config_file,valid_imgs)

   logger.info(' %s training batches; %s validation batches',len(train_gen),len(valid_gen))

   dateString = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d-%H-%M-%S')
   log_path = os.path.join(config_file['tensorboard']['log_dir'],dateString)

   logger.debug('create Adam')
   optimizer = optimizers.Adam(lr=config_file['training']['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

   if args.horovod:
      logger.debug('create horovod optimizer')
      optimizer = hvd.DistributedOptimizer(optimizer)
      if hvd.rank() == 0:
         os.makedirs(log_path)
   else:
      os.makedirs(log_path)
   
   config_proto = create_config_proto(args)
   logger.debug('tf session')
   session = tf.Session(config=config_proto)
   logger.debug('keras backend')
   keras_backend.set_session(session)

   logger.debug('compile model')
   model.compile(loss=loss_func.loss, optimizer=optimizer)
   
   checkpoint = ModelCheckpoint(config_file['model_pars']['model_checkpoint_file'].format(date=dateString),
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=True,
                        mode='min',
                        period=1)

   tensorboard = TB(log_dir=log_path,
                        histogram_freq=config_file['tensorboard']['histogram_freq'],
                        write_graph=config_file['tensorboard']['write_graph'],
                        write_images=config_file['tensorboard']['write_images'],
                        write_grads=config_file['tensorboard']['write_grads'],
                        embeddings_freq=config_file['tensorboard']['embeddings_freq'])

   callbacks = [tensorboard]

   if args.horovod:
      # Horovod: broadcast initial variable states from rank 0 to all other processes.
      # This is necessary to ensure consistent initialization of all workers when
      # training is started with random weights or restored from a checkpoint.
      callbacks.append(bcast_global_variables_op)
      if hvd.rank() == 0:
         callbacks.append(checkpoint)
   else:
      callbacks.append(checkpoint)


   logger.debug('call fit generator')
   model.fit_generator(generator         = train_gen,
                        epochs           = config_file['training']['epochs'],
                        verbose          = config_file['training']['verbose'],
                        validation_data  = valid_gen,
                        callbacks        = callbacks,
                        workers          = 1,
                        max_queue_size   = 5,
                        steps_per_epoch  = len(train_gen),
                        validation_steps = len(valid_gen))
   





if __name__ == "__main__":
   print('start main')
   main()
