#!/usr/bin/env python
import os,argparse,logging,json,glob,datetime
import numpy as np
import yolo_model
from BatchGenerator import BatchGenerator,BatchGeneratorGroup
import loss_func

from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras import backend as keras_backend
from callbacks import TB

import tensorflow as tf

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
   parser.add_argument('--tb_logdir', '-l',
                       help='tensorboard logdir for this job.',default=None)
   parser.add_argument('--horovod', default=False,
                       help='use Horovod',action='store_true')
   parser.add_argument('--ml_comm', default=False,
                       help='use Cray PE ML Plugin',action='store_true')
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
   parser.add_argument('--batchgroup',action='store_true',default=False,
                       help='use subprocess group batchgroup instead of standard BatchGenerator')
   parser.add_argument('--batchgroup_size',type=int,default=4,
                       help='number of subprocesses in the batchgroup')
   parser.add_argument('--batch_queue_size',type=int,default=4,
                       help='number of batch queues in the fit_generator')
   parser.add_argument('--batch_queue_workers',type=int,default=0,
                       help='number of batch workers in the fit_generator')
   parser.add_argument('--timeline',action='store_true',default=False,
                       help='enable chrome timeline profiling')
   parser.add_argument('--timeline_filename',default='timeline_profile.json',
                       help='filename to use for timeline json data')

   args = parser.parse_args()


   log_level = logging.DEBUG
   if args.horovod:
      print("importing hvd")
      import horovod.keras as hvd
      print('horovod from: %s' % hvd.__file__)
      print('hvd init')
      hvd.init()
      print("Rank:",hvd.rank())
      if hvd.rank() > 0:
         log_level = logging.WARNING
      logging.basicConfig(level=log_level,format='%(asctime)s %(levelname)s:' + '{:05d}'.format(hvd.rank()) + ':%(name)s:%(process)s:%(thread)s:%(message)s')
    if self.args.ml_comm:
        logger.debug("importing ml_comm")
        import ml_comm as mc
        from plugin_keras import InitPluginCallback, BroadcastVariablesCallback, DistributedOptimizer
        self.mc = mc
        self.InitPluginCallback = InitPluginCallback
        self.BroadcastVariablesCallback = BroadcastVariablesCallback
        self.DistributedOptimizer = DistributedOptimizer
        logger.info('ml_comm from: %s',mc.__file__)
        logger.debug('mc init')
        self.mc.init_mpi()
        logger.info("Rank: %s",self.mc.get_rank())
        self.rank = self.mc.get_rank()
        self.nranks = self.mc.get_nranks()
   else:
      logging.basicConfig(level=log_level,format='%(asctime)s %(levelname)s:%(name)s:%(process)s:%(thread)s:%(message)s')

   logger.info('config_file:           %s',args.config_file)
   logger.info('tb_logdir:             %s',args.tb_logdir)
   logger.info('horovod:               %s',args.horovod)
   logger.info('num_files:             %s',args.num_files)
   logger.info('lr:                    %s',args.lr)
   logger.info('num_intra:             %s',args.num_intra)
   logger.info('kmp_blocktime:         %s',args.kmp_blocktime)
   logger.info('kmp_affinity:          %s',args.kmp_affinity)
   logger.info('batchgroup:            %s',args.batchgroup)
   logger.info('batchgroup_size:       %s',args.batchgroup_size)
   logger.info('batch_queue_size:      %s',args.batch_queue_size)
   logger.info('batch_queue_workers:   %s',args.batch_queue_workers)
   logger.info('timeline:              %s',args.timeline)
   logger.info('timeline_filename:     %s',args.timeline_filename)

   if args.timeline:
      from tensorflow.python.client import timeline
   
   logger.debug('create config proto')
   config_proto = create_config_proto(args)
   keras_backend.set_session(tf.Session(config=config_proto))

   # load configuration
   config_file = json.load(open(args.config_file))

   
   # build model
   model = yolo_model.build_model(config_file)


   # get inputs
   train_gen,valid_gen = get_image_generators(config_file,args)

   # pass configuration to loss function
   loss_func.set_config(config_file)

   # set learning rate
   if args.horovod:
      learning_rate = config_file['training']['learning_rate'] * hvd.size()
   else:
      learning_rate = config_file['training']['learning_rate']
   
   # create optmization function
   logger.debug('create Adam')
   optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

   if args.horovod:
      logger.debug('create horovod optimizer')
      optimizer = hvd.DistributedOptimizer(optimizer)

   logger.debug('compile model')
   if args.timeline:
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      model.compile(loss=loss_func.loss2, optimizer=optimizer, options=run_options, run_metadata=run_metadata)
   else:
      model.compile(loss=loss_func.loss2, optimizer=optimizer)
   
   # create checkpoint callback
   dateString = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d-%H-%M-%S')
   
   # create log path for tensorboard
   log_path = os.path.join(config_file['tensorboard']['log_dir'],dateString)
   if args.tb_logdir is not None:
      log_path = args.tb_logdir
   

   callbacks = []
   

   verbose = config_file['training']['verbose']
   if args.horovod:
      
      # Horovod: broadcast initial variable states from rank 0 to all other processes.
      # This is necessary to ensure consistent initialization of all workers when
      # training is started with random weights or restored from a checkpoint.
      callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
      
      # Horovod: average metrics among workers at the end of every epoch.
      #
      # Note: This callback must be in the list before the ReduceLROnPlateau,
      # TensorBoard or other metrics-based callbacks.
      callbacks.append(hvd.callbacks.MetricAverageCallback())

      # create tensorboard callback
      tensorboard = TB(log_dir=log_path,
                     histogram_freq=config_file['tensorboard']['histogram_freq'],
                     write_graph=config_file['tensorboard']['write_graph'],
                     write_images=config_file['tensorboard']['write_images'],
                     write_grads=config_file['tensorboard']['write_grads'],
                     embeddings_freq=config_file['tensorboard']['embeddings_freq'])
      callbacks.append(tensorboard)
      
      if hvd.rank() == 0:
         verbose = config_file['training']['verbose']
         os.makedirs(log_path)

         checkpoint = ModelCheckpoint(config_file['model_pars']['model_checkpoint_file'].format(date=dateString),
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=True,
                        mode='min',
                        period=1)
         callbacks.append(checkpoint)

         
      else:
         verbose = 0
   else:
      os.makedirs(log_path)

      checkpoint = ModelCheckpoint(config_file['model_pars']['model_checkpoint_file'].format(date=dateString),
                     monitor='val_loss',
                     verbose=1,
                     save_best_only=True,
                     mode='min',
                     period=1)
      callbacks.append(checkpoint)


   logger.debug('callbacks: %s',callbacks)


   logger.debug('call fit generator')
   model.fit_generator(generator         = train_gen,
                        epochs           = config_file['training']['epochs'],
                        verbose          = verbose,
                        validation_data  = valid_gen,
                        callbacks        = callbacks,
                        workers          = args.batch_queue_workers,
                        max_queue_size   = args.batch_queue_size,
                        steps_per_epoch  = len(train_gen),
                        validation_steps = config_file['training']['steps_per_valid'],
                        use_multiprocessing=False)
   logger.debug('done fit gen')

   if args.batchgroup:
      train_gen.exit()
      valid_gen.exit()

   if args.timeline:
      logger.info('output timeline profile')
      trace = timeline.Timeline(step_stats=run_metadata.step_stats)
      with open(args.timeline_filename, 'w') as f:
         f.write(trace.generate_chrome_trace_format())

   logger.info('done')
   

def get_image_generators(config_file,args):
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

   train_imgs = filelist[:train_file_index]
   if args.batchgroup:
      train_gen = BatchGeneratorGroup(config_file,train_imgs,workers=args.batchgroup_size,name='train')
   else:
      train_gen = BatchGenerator(config_file,train_imgs,name='train')
   valid_imgs = filelist[train_file_index:nfiles]
   if args.batchgroup:
      valid_gen = BatchGeneratorGroup(config_file,valid_imgs,workers=args.batchgroup_size,name='valid')
   else:
      valid_gen = BatchGenerator(config_file,valid_imgs,name='valid')

   logger.info(' %s training batches; %s validation batches',len(train_gen),len(valid_gen))

   return train_gen,valid_gen


if __name__ == "__main__":
   print('start main')
   main()
