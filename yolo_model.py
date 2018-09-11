import logging
from keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization, Reshape
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate

logger = logging.getLogger(__name__)


def build_model(config,print_summary=True):
   
   input_image = Input(shape=tuple(config['data_handling']['image_shape']))
   output = input_image

   layer_num = 0

   # 2 layers with pooling
   for conf in [[32,(3,3),(2,2)],
                [64,(3,3),(1,2)],
                [128,(3,3),(1,2)],
                [128,(3,3),(1,2)],
                [128,(3,3),(2,2)],]:
      output = CBLP_layer(output,
               filters=conf[0],
               window=conf[1],
               pool_size=conf[2],
               layer_num = layer_num,
         )
      layer_num += 1

   # layers without pooling
   for conf in [
               [64,(1,1)],
               [128,(3,3)],
               [256,(3,3)],
               ]:
      output = CBL_layer(output,
               filters=conf[0],
               window=conf[1],
               layer_num = layer_num,
         )
      layer_num += 1

   # capture skip connection
   skip_connection = output

   # 4 layers without pooling
   for conf in [
               [512,(3,3)],
               [256,(1,1)],
               [512,(3,3)],
               [1024,(3,3)],
               ]:
      output = CBL_layer(output,
               filters=conf[0],
               window=conf[1],
               layer_num = layer_num,
         )
      layer_num += 1

   output = concatenate([skip_connection,output],axis=1)

   for conf in [[1024,(3,3)]]:
      output = CBLP_layer(output,
               filters=conf[0],
               window=conf[1],
               layer_num = layer_num,
         )
      layer_num += 1

   n_grid_boxes_h,n_grid_boxes_w = output.shape[2:4]
   n_grid_boxes_w = int(str(n_grid_boxes_w))
   n_grid_boxes_h = int(str(n_grid_boxes_h))
   config['training']['gridW'] = n_grid_boxes_w
   config['training']['gridH'] = n_grid_boxes_h

   logger.info('grid size: %s x %s',n_grid_boxes_w,n_grid_boxes_h)

   n_classes = len(config['data_handling']['classes'])
   layer_num += 1
   output = Conv2D(4 + 1 + n_classes,
                        (1,1), strides=(1,1),
                        padding='same',
                        name='DetectionLayer_{0}'.format(layer_num),
                        kernel_initializer='lecun_normal',
                        data_format='channels_first')(output)
   output = Reshape((n_grid_boxes_h, n_grid_boxes_w, 4 + 1 + n_classes),name='reshape_{0}'.format(layer_num))(output)


   # boxes = Input(shape=(n_grid_boxes_h, n_grid_boxes_w, 4 + 1 + n_classes))
   # output = Lambda(lambda args: args[0],name='lambda_{0}'.format(layer_num))([output, boxes])

   model = Model(input_image,output)

   if print_summary:
      model.summary()

   return model


def CBL_layer(input,
            filters=32,
            window=(3,3),
            strides=(1,1),
            padding='same',
            layer_num=0,
            use_bias=False,
            data_format='channels_first',
            axis=1,
            alpha=0.1,
            ):
   x = Conv2D(filters,window,
            strides=strides,
            padding=padding,
            name='conv2d_{0}'.format(layer_num),
            use_bias=use_bias,
            data_format=data_format)(input)
   x = BatchNormalization(axis=axis,name='norm_{0}'.format(layer_num))(x)
   x = LeakyReLU(alpha=alpha,name='relu_{0}'.format(layer_num))(x)
   return x


def CBLP_layer(input,
            filters=32,
            window=(3,3),
            strides=(1,1),
            padding='same',
            layer_num=0,
            use_bias=False,
            data_format='channels_first',
            axis=1,
            alpha=0.1,
            pool_size=(2,2),
            ):
   x = CBL_layer(input,
            filters=filters,
            window=window,
            strides=strides,
            padding=padding,
            layer_num=layer_num,
            use_bias=use_bias,
            data_format=data_format,
            axis=axis,
            alpha=alpha,
            )
   x = MaxPooling2D(pool_size=pool_size, data_format=data_format,name='pool_{0}'.format(layer_num))(x)
   return x
