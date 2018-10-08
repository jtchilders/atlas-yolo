from keras.utils import Sequence
import numpy as np
import logging,multiprocessing,time

logger = logging.getLogger(__name__)

# Right now the truth are format as:
# bool objectFound         # if truth particle is in the hard scattering process
# bbox x                   # globe eta
# bbox y                   # globe phi
# bbox width               # Gaussian sigma (required to be 3*sigma<pi)
# bbox height              # Gaussian sigma (required to be 3*sigma<pi)
# bool class1              # truth u/d
# bool class2              # truth s
# bool class3              # truth c
# bool class4              # truth b
# bool class_other         # truth g

# indices in 'truth' vector
BBOX_CENTER_X = 1
BBOX_CENTER_Y = 2
BBOX_WIDTH = 3
BBOX_HEIGHT = 4
CLASS_START = 5
CLASS_END = 10


class BatchGenerator(Sequence):
   def __init__(self,config,filelist,name='',truth_passthrough=False):

      self.config          = config
      self.name            = name
      self.truth_passthrough = truth_passthrough

      # get file list
      self.filelist        = filelist
      logger.info('%s: found %s input files',self.name,len(self.filelist))
      if len(self.filelist) < 1:
         raise Exception('%s: length of file list needs to be at least 1' % self.name)

      train_file_index = int(config['data_handling']['training_to_validation_ratio'] * len(self.filelist))
      np.random.shuffle(self.filelist)

      self.train_imgs = self.filelist[:train_file_index]
      self.valid_imgs = self.filelist[train_file_index:]

      self.shuffle = self.config['data_handling']['shuffle']

      self.n_chan,self.img_h,self.img_w = tuple(config['data_handling']['image_shape'])
   

      self.evts_per_file      = config['data_handling']['evt_per_file']
      self.nevts              = len(self.filelist) * self.evts_per_file
      self.batch_size         = config['training']['batch_size']
      self.max_box_per_image  = config['model_pars']['max_box_per_image']
      self.n_grid_boxes_w     = config['training']['gridW']
      self.n_grid_boxes_h     = config['training']['gridH']
      self.n_classes          = len(config['data_handling']['classes'])

      self.pixels_per_grid_w  = float(self.img_w) / self.n_grid_boxes_w
      self.pixels_per_grid_h  = float(self.img_h) / self.n_grid_boxes_h

      self.obj_multiplier     = [1.,
                                 1. / self.pixels_per_grid_w,
                                 1. / self.pixels_per_grid_h,
                                 1. / self.pixels_per_grid_w,
                                 1. / self.pixels_per_grid_h]
      self.obj_multiplier     += [1. for _ in range(self.n_classes)]
      self.obj_multiplier     = np.array(self.obj_multiplier)

      self.current_file_index = -1
      self.images = None
      self.truth_boxes = None


      logger.debug('%s: evts_per_file:           %s',self.name,self.evts_per_file)
      logger.debug('%s: nevts:                   %s',self.name,self.nevts)
      logger.debug('%s: batch_size:              %s',self.name,self.batch_size)
      logger.debug('%s: max_box_per_image:       %s',self.name,self.max_box_per_image)
      logger.debug('%s: n_grid_boxes_w:          %s',self.name,self.n_grid_boxes_w)
      logger.debug('%s: n_grid_boxes_h:          %s',self.name,self.n_grid_boxes_h)
      logger.debug('%s: n_classes:               %s',self.name,self.n_classes)
      logger.debug('%s: pixels_per_grid_w:       %s',self.name,self.pixels_per_grid_w)
      logger.debug('%s: pixels_per_grid_h:       %s',self.name,self.pixels_per_grid_h)


   def __len__(self):
      return int(float(self.nevts) / self.batch_size)

   def num_classes(self):
      return len(self.config['data_handling']['classes'])

   def size(self):
      return self.nevts

   # return a batch of images starting at the given index
   def __getitem__(self, idx):

      try:
         start = time.time()
         logger.debug('%s: starting get batch',self.name)

         # count the number of images
         instance_count = 0

         ##########
         # prepare output variables:

         # input images
         x_batch = np.zeros((self.batch_size, self.n_chan, self.img_h, self.img_w))
         # list of boxes
         # b_batch = np.zeros((self.batch_size, self.n_grid_boxes_h,
         #                     self.n_grid_boxes_w, 4 + 1 + self.n_classes))
         # desired network output
         if not self.truth_passthrough:
            y_batch = np.zeros((self.batch_size, self.n_grid_boxes_h,
                             self.n_grid_boxes_w, 4 + 1 + self.n_classes))
         else:
            y_batch = np.zeros((self.batch_size,4+1+self.n_classes))

         ##########
         # calculate which file needed based on list of files, events per file,
         # ad which batch index

         epoch_image_index = idx * self.batch_size

         file_index = int(epoch_image_index / self.evts_per_file)
         if file_index >= len(self.filelist):
            raise Exception('{0}: file_index {1} is outside range for filelist {2}'.format(self.name,file_index,len(self.filelist)))
         
         image_index = epoch_image_index % self.evts_per_file
         
         logger.debug('%s: opening file with idx %s file_index %s image_index %s',self.name,
               idx,
               file_index,image_index)

         ######
         # open the file
         if self.current_file_index != file_index or self.images is None:
            logger.debug('%s: new file opening %s %s',self.name,self.current_file_index,file_index)
            self.current_file_index = file_index
            file_content = np.load(self.filelist[self.current_file_index])
            self.images = file_content['raw']
            self.truth_boxes = file_content['truth']
            logger.debug('%s: shape images %s truth %s',self.name,self.images.shape,self.truth_boxes.shape)
         else:
            logger.debug('%s: not opening file  %s %s',self.name,self.current_file_index,file_index)
         
            

         ########
         # loop over the batch size
         # create the outputs
         for i in range(self.batch_size):
            logger.debug('%s: loop %s start',self.name,i)

            # if our image index has gone past the number
            # of images per file, then open the next file
            if image_index >= self.evts_per_file:
               logger.debug('%s: new file opening %s',self.name,self.current_file_index)
               self.current_file_index += 1
               
               if self.current_file_index >= len(self.filelist):
                  self.on_epoch_end()
                  self.current_file_index = 0

               file_content = np.load(self.filelist[self.current_file_index])
               self.images = file_content['raw']
               self.truth_boxes = file_content['truth']
               logger.debug('%s: shape images %s truth %s',self.name,self.images.shape,self.truth_boxes.shape)
               image_index = 0

            logger.debug('%s: image_index = %s  file_index = %s',self.name,image_index,self.current_file_index)

            # get the image and boxes
            x_batch[instance_count] = self.images[image_index]
            all_objs = self.truth_boxes[image_index]
            obj = all_objs[0]
            logger.warning('%s: current batch generator only supports 1 object per image',self.name)

            logger.debug('%s: loop %s file loaded',self.name,i)

            # construct output from object's x, y, w, h
            # true_box_index = 0
            
            # for obj in all_objs:

            # convert pixel coords to grid coords
            # for center x/y:
            # center y in pixel coords (200)
            # convert to grid coords (200/ (256/16)) = 12.5
            # center x in pixel coords (200)
            # convert to grid coords (200/ (9600/150)) = 3.125
            # for width/height:
            # width in pixel coords (100)
            # convert to grid coords (100/ (9600/150)) = 1.5
            # height in pixel coords (30)
            # convert to grid coords (30 / (256/16)) = 1.8

            logger.debug('%s: obj = %s',self.name,obj)
            obj = np.multiply(obj,self.obj_multiplier)

            # truncate to get grid position (12,3)
            grid_x = int(obj[BBOX_CENTER_X])
            grid_y = int(obj[BBOX_CENTER_Y])

            # convert x/y to internal grid coords
            # for y 12.5 -> 0.5
            # for x 3.125 -> 0.125
            sub = np.zeros(obj.shape)
            sub[BBOX_CENTER_X] = grid_x
            sub[BBOX_CENTER_Y] = grid_y
            obj = obj - sub

            logger.debug('%s: new obj = %s',self.name,obj)

            # assign ground truth x, y, w, h, confidence and class probs to y_batch
            if not self.truth_passthrough:
               y_batch[instance_count][grid_y][grid_x] = obj
            else:
               y_batch[instance_count] = self.truth_boxes[image_index]

            # assign the true box to b_batch
            # b_batch[instance_count, grid_y, grid_x, 0:4] = box

            # true_box_index += 1
            # true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

            logger.debug('%s: loop %s images converted',self.name,i)
                            
            # assign input image to x_batch
            # if self.norm is not None:
            #    x_batch[instance_count] = self.norm(img)
            
            

            # increase instance counter in current batch
            instance_count += 1
            image_index += 1

         logger.debug('%s: x_batch = %s',self.name,np.sum(x_batch))
         logger.debug('%s: y_batch = %s',self.name,np.sum(y_batch))
         logger.debug('%s: x_batch shape = %s',self.name,x_batch.shape)
         logger.debug('%s: y_batch shape = %s',self.name,y_batch.shape)
         logger.debug('%s: exiting getitem duration: %s, file_index = %s',self.name,(time.time() - start),self.current_file_index)

         # print(' new batch created', idx)

         return x_batch, y_batch
      except Exception as e:
         logger.exception('%s: caught exception %s',self.name,str(e))
         raise

   def on_epoch_end(self):
      if self.shuffle:
         np.random.shuffle(self.filelist)


class BatchGeneratorGroup(Sequence):

   def __init__(self,config,filelist,workers=1,name=''):

      self.config          = config
      self.workers         = workers
      self.name            = name
      self.batch_size         = config['training']['batch_size']

      self.dispatcher = DistributedBatcher(filelist,self.workers,self.batch_size,self.name)
      self.dispatcher.start()


      
      logger.info('found %s input files',len(filelist))
      if len(filelist) < 1:
         raise Exception('length of file list needs to be at least 1')


      self.n_chan,self.img_h,self.img_w = tuple(config['data_handling']['image_shape'])
   

      self.evts_per_file      = config['data_handling']['evt_per_file']
      self.nevts              = len(filelist) * self.evts_per_file
      self.n_grid_boxes_w     = config['training']['gridW']
      self.n_grid_boxes_h     = config['training']['gridH']
      self.n_classes          = len(config['data_handling']['classes'])

      self.pixels_per_grid_w  = float(self.img_w) / self.n_grid_boxes_w
      self.pixels_per_grid_h  = float(self.img_h) / self.n_grid_boxes_h

      self.obj_multiplier     = [1.,
                                 1. / self.pixels_per_grid_w,
                                 1. / self.pixels_per_grid_h,
                                 1. / self.pixels_per_grid_w,
                                 1. / self.pixels_per_grid_h]
      self.obj_multiplier     += [1. for _ in range(self.n_classes)]
      self.obj_multiplier     = np.array(self.obj_multiplier)

      logger.debug('evts_per_file:           %s',self.evts_per_file)
      logger.debug('nevts:                   %s',self.nevts)
      logger.debug('batch_size:              %s',self.batch_size)
      logger.debug('n_grid_boxes_w:          %s',self.n_grid_boxes_w)
      logger.debug('n_grid_boxes_h:          %s',self.n_grid_boxes_h)
      logger.debug('n_classes:               %s',self.n_classes)
      logger.debug('pixels_per_grid_w:       %s',self.pixels_per_grid_w)
      logger.debug('pixels_per_grid_h:       %s',self.pixels_per_grid_h)


   def __len__(self):
      return int(np.ceil(float(self.nevts) / self.batch_size))

   def num_classes(self):
      return len(self.config['data_handling']['classes'])

   def exit(self):
      self.dispatcher.exit()

   def size(self):
      return self.nevts

   # return a batch of images starting at the given index
   def __getitem__(self, idx):
      start = time.time()
      logger.debug('starting get batch index: %s',idx)


      ##########
      # prepare output variables:

      # input images
      x_batch = np.zeros((self.batch_size, self.n_chan, self.img_h, self.img_w))
      # list of boxes
      # b_batch = np.zeros((self.batch_size, self.n_grid_boxes_h,
      #                     self.n_grid_boxes_w, 4 + 1 + self.n_classes))
      # desired network output
      y_batch = np.zeros((self.batch_size, self.n_grid_boxes_h,
                          self.n_grid_boxes_w, 4 + 1 + self.n_classes))


      logger.debug('fill batch size: %s',self.batch_size)
      for i in range(self.batch_size):
         img,truth = self.dispatcher.next()

         x_batch[i] = img

         obj = truth[0]
         logger.warning('current batch generator only supports 1 object per image')

         logger.debug('obj = %s',obj)
         obj = np.multiply(obj,self.obj_multiplier)

         # truncate to get grid position (12,3)
         grid_x = int(obj[BBOX_CENTER_X])
         grid_y = int(obj[BBOX_CENTER_Y])

         # convert x/y to internal grid coords
         # for y 12.5 -> 0.5
         # for x 3.125 -> 0.125
         sub = np.zeros(obj.shape)
         sub[BBOX_CENTER_X] = grid_x
         sub[BBOX_CENTER_Y] = grid_y
         obj = obj - sub

         logger.debug('new obj = %s',obj)

         # assign ground truth x, y, w, h, confidence and class probs to y_batch
         y_batch[i][grid_y][grid_x] = obj


      logger.debug('x_batch = %s',np.sum(x_batch))
      logger.debug('y_batch = %s',np.sum(y_batch))
      logger.debug('x_batch shape = %s',x_batch.shape)
      logger.debug('y_batch shape = %s',y_batch.shape)
      logger.debug('exiting run time = %s',(time.time() - start))

      # print(' new batch created', idx)

      return x_batch, y_batch


class EventFileGenerator(object):
   def __init__(self,filename):
      start = time.time()
      logger.debug('EventFileGenerator: opening file %s',filename)
      self.filename = filename
      self.filedata = np.load(filename)
      self.entry = 0
      self.nentries = len(self.filedata['truth'])
      logger.debug('EventFileGenerator: done opening file in %s',(time.time() - start))

   def __iter__(self):
      return self

   def __next__(self):
      return self.next()

   def next(self):
      start = time.time()
      if self.entry < self.nentries:
         img = self.filedata['raw'][self.entry]
         truth = self.filedata['truth'][self.entry]
         self.entry += 1
         logger.debug('returning event from filename %s in %s',self.filename,(time.time() - start))
         return img,truth
      else:
         raise StopIteration()


class EventProviderProcess(multiprocessing.Process):
   def __init__(self,filelist,name=''):
      super(EventProviderProcess,self).__init__()
      self.filelist = filelist
      self.name = name

      self.output_queue = multiprocessing.Queue()

      self.exit_flag = multiprocessing.Event()

   def exit(self):
      self.exit_flag.set()

   def get_event(self):
      if not self.output_queue.empty():
         logger.debug('EventProviderProcess/%s: return event from queue ',self.name)
         return self.output_queue.get()
      else:
         logger.debug('EventProviderProcess/%s: event queue empty',self.name)
         return None

   def run(self):
      logger.debug('EventProviderProcess/%s: starting with %s files',self.name,len(self.filelist))
      while not self.exit_flag.is_set():
         logger.debug('EventProviderProcess/%s: start new epoch ',self.name)

         for filename in self.filelist:
            logger.debug('EventProviderProcess/%s: opening file %s',self.name,filename)
            
            evtfile = EventFileGenerator(filename)

            for output in evtfile:
               logger.debug('EventProviderProcess/%s: place event in queue, qsize = %s',self.name.self.output_queue.qsize())
               self.output_queue.put(output)

               # wait until queue is empty
               while not self.output_queue.empty() and not self.exit_flag.is_set():
                  time.sleep(1)

               if self.exit_flag.is_set():
                  logger.info('EventProviderProcess/%s: batcher exiting',self.name)
                  return

         logger.debug('EventProviderProcess/%s: shuffling filelist ',self.name)
         np.random.shuffle(self.filelist)

      logger.info('EventProviderProcess/%s: batcher exiting',self.name)


class DistributedBatcher(multiprocessing.Process):

   def __init__(self,filelist,workers,batch_size,name):
      super(DistributedBatcher,self).__init__()

      logger.debug('DistributedBatcher/%s: create batcher filelists from %s files',name,len(filelist))
      self.name = name
      self.batch_size = batch_size

      if len(filelist) < workers:
         logger.info('DistributedBatcher/%s: tried to create more workers than files so restricting workers to %s',self.name,len(filelist))
         self.workers = len(filelist)
      else:
         self.workers = workers

      files_per_batcher = int(len(filelist) / self.workers)

      logger.debug('DistributedBatcher/%s: files per batcher = %s',self.name,files_per_batcher)
      self.batcher_filelists = []
      for i in range(self.workers):
         self.batcher_filelists.append(filelist[i * files_per_batcher:(i + 1) * files_per_batcher])

      leftover_files = filelist[files_per_batcher * self.workers:0]
      logger.debug('DistributedBatcher/%s: %s leftovers',self.name,len(leftover_files))

      for i in range(len(leftover_files)):
         self.batcher_filelists[i % self.workers].append(leftover_files[i])


      self.output_queue = multiprocessing.Queue()

      self.exit_flag = multiprocessing.Event()

   def exit(self):
      self.exit_flag.set()

   def next(self):
      logger.debug('DistributedBatcher/%s:  next',self.name)
      output = None
      while output is None:
         output = self.output_queue.get()
         if output is None:
            time.sleep(1)
      return output


   def run(self):
      
      logger.debug('DistributedBatcher/%s: create %s batchers',self.name,self.workers)
      batchers = []
      for i in range(self.workers):
         logger.debug('DistributedBatcher/%s: batcher %s gets %s files',self.name,i,len(self.batcher_filelists[i]))
         batcher = EventProviderProcess(self.batcher_filelists[i],'%s_%02d' % (self.name,i))
         batcher.start()
         batchers.append(batcher)

      while not self.exit_flag.is_set():

         if self.output_queue.qsize() < self.batch_size:
            start = time.time()
            i = np.random.random_integers(0,self.workers - 1)
            logger.debug('DistributedBatcher/%s:  adding file to queue from batcher %i',self.name,i)
            # add event to output queue
            i = np.random.random_integers(0,self.workers - 1)
            output = None
            while output is None:
               output = batchers[i].get_event()
               if output is None:
                  time.sleep(1)
                  if self.exit_flag.is_set():
                     logger.debug('DistributedBatcher/%s:  exiting',self.name)
                     for batcher in batchers:
                        batcher.exit()
                     for batcher in batchers:
                        batcher.join()
                     return
            self.output_queue.put(output)
            logger.debug('DistributedBatcher/%s: time to add event = %s',self.name,(time.time() - start))
         else:
            time.sleep(1)


      for batcher in batchers:
         batcher.exit()
      for batcher in batchers:
         batcher.join()
      logger.debug('DistributedBatcher/%s:  exiting',self.name)





def global_to_grid(x,y,w,h,num_grid_x,num_grid_y):
   ''' convert global bounding box coords to
   grid coords. x,y = box center in relative coords
   going from 0 to 1. w,h are the box width and
   height in relative coords going from 0 to 1.
   num_grid_x,num_grid_y define the number of bins
   in x and y for the grid.
   '''
   global_coords  = [x,y]
   global_sizes   = [w,h]
   num_grid_bins  = [num_grid_x,num_grid_y]
   grid_coords    = [0.,0.]
   grid_sizes     = [0.,0.]
   for i in range(len(global_coords)):
      grid_bin_size = 1. / num_grid_bins[i]
      grid_bin = int(global_coords[i] / grid_bin_size)
      grid_coords[i] = (global_coords[i] - grid_bin_size * grid_bin) / grid_bin_size
      grid_sizes[i] = (global_sizes[i] / grid_bin_size)

   return grid_coords,grid_sizes


def grid_to_global(grid_bin_x, grid_bin_y,grid_x,grid_y,grid_w,grid_h,num_grid_x,num_grid_y):
   ''' inverse of global_to_grd '''
   grid_bins      = [grid_bin_x,grid_bin_y]
   grid_coords    = [grid_x,grid_y]
   grid_sizes     = [grid_w,grid_h]
   num_grid_bins  = [num_grid_x,num_grid_y]
   global_coords  = [0.,0.]
   global_sizes   = [0.,0.]
   for i in range(len(global_coords)):
      grid_bin_size     = 1. / num_grid_bins[i]
      global_coords[i]  = (grid_bins[i] + grid_coords[i]) * grid_bin_size
      global_sizes[i]   = grid_sizes[i] * grid_bin_size

   return global_coords,global_sizes
