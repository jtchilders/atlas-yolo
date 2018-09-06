#!/usr/bin/env python
import os,sys,optparse,logging,json,glob,datetime
import yolo_model
from BatchGenerator import BatchGenerator
import loss_func
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from callbacks import TB
import numpy as np
logger = logging.getLogger(__name__)


def main():
   ''' simple starter program that can be copied for use when starting a new script. '''
   logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s:%(name)s:%(thread)s:%(message)s')

   parser = optparse.OptionParser(description='')
   parser.add_option('-c','--config',dest='config',help='configuration in standard json format.')
   opts,args = parser.parse_args()

   
   mandatory_args = [
                     'config',
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

   train_file_index = int(config['data_handling']['training_to_validation_ratio'] * len(filelist))
   np.random.shuffle(filelist)

   model = yolo_model.build_model(config)
   model.summary()

   logger.info('grid (w,h) = (%s,%s)',config['training']['gridW'],config['training']['gridH'])
   loss_func.set_config(config)

   train_imgs = filelist[:train_file_index]
   train_gen = BatchGenerator(config,train_imgs)
   valid_imgs = filelist[train_file_index:]
   valid_gen = BatchGenerator(config,valid_imgs)

   optimizer = optimizers.Adam(lr=config['training']['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

   model.compile(loss=loss_func.loss, optimizer=optimizer)
   # loss_func.set_config(config)
   # model.compile(loss=loss_func.loss, optimizer=optimizer)
   dateString = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d-%H-%M-%S')
   log_path = os.path.join(config['tensorboard']['log_dir'],dateString)
   os.makedirs(log_path)
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

   model.fit_generator(generator         = train_gen,
                        epochs           = config['training']['epochs'],
                        verbose          = config['training']['verbose'],
                        validation_data  = valid_gen,
                        callbacks        = [checkpoint, tensorboard],
                        workers          = 1,
                        max_queue_size   = 5)
   





if __name__ == "__main__":
   main()
