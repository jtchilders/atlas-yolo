#!/usr/bin/env python
import os,sys,optparse,logging,json,glob,datetime
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

config = tf.ConfigProto()
config.intra_op_parallelism_threads = int(os.environ['TF_INTRA_THREADS']) if 'TF_INTRA_THREADS' in os.environ else None
config.inter_op_parallelism_threads = int(os.environ['TF_INTER_THREADS']) if 'TF_INTER_THREADS' in os.environ else None
config.allow_soft_placement         = True
session = tf.Session(config=config)
keras_backend.set_session(session)


def main():
   ''' simple starter program that can be copied for use when starting a new script. '''

   parser = optparse.OptionParser(description='')
   parser.add_option('-c','--config',dest='config',help='configuration in standard json format.')
   parser.add_option('-a','--horovod',dest='horovod',help='turn on horovod.',default=False,action='store_true')
   parser.add_option('-n','--num_files',dest='num_files',help='limit the number of files to process. default is all',default=-1,type='int')
   opts,args = parser.parse_args()


   if opts.horovod:
      import horovod.keras as hvd
      hvd.init()
      logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s:' + '{:05d}'.format(hvd.rank()) + ':%(name)s:%(thread)s:%(message)s')
   else:
      logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s:%(name)s:%(thread)s:%(message)s')
   
   mandatory_args = [
                     'config',
                     'horovod',
                  ]

   for man in mandatory_args:
      if man not in opts.__dict__ or opts.__dict__[man] is None:
         logger.error('Must specify option: ' + man)
         parser.print_help()
         sys.exit(-1)


   # load configuration
   config = json.load(open(opts.config))

   # get file list
   filelist = glob.glob(config['data_handling']['input_file_glob'])
   logger.info('found %s input files',len(filelist))
   if len(filelist) < 2:
      raise Exception('length of file list needs to be at least 2 to have train & validate samples')

   nfiles = len(filelist)
   if opts.num_files > 0:
      nfiles = opts.num_files

   train_file_index = int(config['data_handling']['training_to_validation_ratio'] * nfiles)
   np.random.shuffle(filelist)

   model = yolo_model.build_model(config)
   model.summary()

   logger.info('grid (w,h) = (%s,%s)',config['training']['gridW'],config['training']['gridH'])
   loss_func.set_config(config)

   train_imgs = filelist[:train_file_index]
   train_gen = BatchGenerator(config,train_imgs)
   valid_imgs = filelist[train_file_index:nfiles]
   valid_gen = BatchGenerator(config,valid_imgs)

   logger.info(' %s training batches; %s validation batches',len(train_gen),len(valid_gen))

   dateString = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d-%H-%M-%S')
   log_path = os.path.join(config['tensorboard']['log_dir'],dateString)

   logger.debug('create Adam')
   optimizer = optimizers.Adam(lr=config['training']['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

   if opts.horovod:
      logger.debug('create horovod optimizer')
      optimizer = hvd.DistributedOptimizer(optimizer)
      if hvd.rank() == 0:
         os.makedirs(log_path)
   else:
      os.makedirs(log_path)

   logger.debug('compile model')
   model.compile(loss=loss_func.loss, optimizer=optimizer)
   
   checkpoint = ModelCheckpoint(config['model_pars']['model_checkpoint_file'].format(date=dateString),
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=True,
                        mode='min',
                        period=1)

   tensorboard = TB(log_dir=log_path,
                        histogram_freq=config['tensorboard']['histogram_freq'],
                        write_graph=config['tensorboard']['write_graph'],
                        write_images=config['tensorboard']['write_images'],
                        write_grads=config['tensorboard']['write_grads'],
                        embeddings_freq=config['tensorboard']['embeddings_freq'])

   callbacks = [tensorboard]

   if opts.horovod:
      # Horovod: broadcast initial variable states from rank 0 to all other processes.
      # This is necessary to ensure consistent initialization of all workers when
      # training is started with random weights or restored from a checkpoint.
      callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
      if hvd.rank() == 0:
         callbacks.append(checkpoint)
   else:
      callbacks.append(checkpoint)


   logger.debug('call fit generator')
   model.fit_generator(generator         = train_gen,
                        epochs           = config['training']['epochs'],
                        verbose          = config['training']['verbose'],
                        validation_data  = valid_gen,
                        callbacks        = callbacks,
                        workers          = 1,
                        max_queue_size   = 5,
                        steps_per_epoch  = len(train_gen),
                        validation_steps = len(valid_gen))
   





if __name__ == "__main__":
   main()
