from keras.utils import Sequence
import numpy as np
import logging

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
   def __init__(self,config,filelist):

      self.config          = config

      # get file list
      self.filelist        = filelist
      logger.info('found %s input files',len(self.filelist))
      if len(self.filelist) < 2:
         raise Exception('length of file list needs to be at least 2 to have train & validate samples')

      train_file_index = int(config['data_handling']['training_to_validation_ratio'] * len(self.filelist))
      np.random.shuffle(self.filelist)

      self.train_imgs = self.filelist[:train_file_index]
      self.valid_imgs = self.filelist[train_file_index:]

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

      logger.debug('evts_per_file:           %s',self.evts_per_file)
      logger.debug('nevts:                   %s',self.nevts)
      logger.debug('batch_size:              %s',self.batch_size)
      logger.debug('max_box_per_image:       %s',self.max_box_per_image)
      logger.debug('n_grid_boxes_w:          %s',self.n_grid_boxes_w)
      logger.debug('n_grid_boxes_h:          %s',self.n_grid_boxes_h)
      logger.debug('n_classes:               %s',self.n_classes)
      logger.debug('pixels_per_grid_w:       %s',self.pixels_per_grid_w)
      logger.debug('pixels_per_grid_h:       %s',self.pixels_per_grid_h)


   def __len__(self):
      return int(np.ceil(float(self.nevts) / self.batch_size))

   def num_classes(self):
      return len(self.config['data_handling']['classes'])

   def size(self):
      return self.nevts

   # return a batch of images starting at the given index
   def __getitem__(self, idx):
      logger.debug('starting get batch')

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
      y_batch = np.zeros((self.batch_size, self.n_grid_boxes_h,
                          self.n_grid_boxes_w, 4 + 1 + self.n_classes))

      ##########
      # calculate which file needed based on list of files, events per file,
      # ad which batch index
      file_index = int(idx / self.evts_per_file)
      image_index = idx % self.evts_per_file
      
      logger.debug('opening file with idx %s file_index %s image_index %s',
            idx,
            file_index,image_index)

      ######
      # open the file
      if file_index < len(self.filelist):
         file_content = np.load(self.filelist[file_index])
         images = file_content['raw']
         truth_boxes = file_content['truth']
      else:
         raise Exception('file_index {0} is outside range for filelist {1}'.format(file_index,len(self.filelist)))

      ########
      # loop over the batch size
      # create the outputs
      for i in range(self.batch_size):
         logger.debug('loop %s start',i)

         # if our image index has gone past the number
         # of images per file, then open the next file
         if image_index >= self.evts_per_file:
            file_index += 1
            file_content = np.load(self.filelist[file_index])
            images = file_content['raw']
            truth_boxes = file_content['truth']
            image_index = 0

         # get the image and boxes
         x_batch[instance_count] = images[image_index]
         all_objs = truth_boxes[image_index]
         obj = all_objs[0]
         logger.warning('current batch generator only supports 1 object per image')

         logger.debug('loop %s file loaded',i)

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
         y_batch[instance_count][grid_y][grid_x] = obj

         # assign the true box to b_batch
         # b_batch[instance_count, grid_y, grid_x, 0:4] = box

         # true_box_index += 1
         # true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

         logger.debug('loop %s images converted',i)
                         
         # assign input image to x_batch
         # if self.norm is not None:
         #    x_batch[instance_count] = self.norm(img)
         
         

         # increase instance counter in current batch
         instance_count += 1
         image_index += 1

      logger.debug('x_batch = %s',np.sum(x_batch))
      logger.debug('y_batch = %s',np.sum(y_batch))
      logger.debug('exiting')

      # print(' new batch created', idx)

      return x_batch, y_batch

   def on_epoch_end(self):
      if self.shuffle:
         np.random.shuffle(self.filelist)

    


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
