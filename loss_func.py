from keras import backend as K

using_tf = False
if 'tensorflow' in K.backend():
   import tensorflow as tf
   using_tf = True

config = None


def set_config(local_config):
   global config
   config = local_config


def loss(y_true, y_pred):

   true_sum = K.sum(y_true)
   pred_sum = K.sum(y_pred)

   loss = y_pred - y_true
   loss = K.square(loss)

   if using_tf:
      loss = tf.Print(loss,[tf.timestamp()],'time since epoch',-1,100)
      loss = tf.Print(loss,[true_sum,pred_sum],'sum true/pred = ',-1,100)
      loss = tf.Print(loss,[y_true],'y_true = ',-1,100)

      loss = tf.Print(loss,[y_pred],'y_pred = ',-1,100)

      loss = tf.Print(loss,[loss],'loss = ',-1,100)


   return loss


def loss2(y_true,y_pred):

   pass
