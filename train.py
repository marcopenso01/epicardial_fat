import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model

def selective_kernel(input1, input2, channel, ratio=8):
    '''
    input1: input tensor from the 2D network (x,y,channels)
    input2: input tensor from the 3D network (x,y,depth,channels)
    channel: channel number of the result
    return: processed tensor
    '''   
    inputs_shape = tf.shape(input1)
    b, h, w = inputs_shape[0], inputs_shape[1], inputs_shape[2]
    xs = []
    xs.append(input1)

    conv2 = ReLU()(BatchNormalization()(Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input2)))
    conv2 = tf.keras.backend.squeeze(conv2, axis=-1)
    conv2 = ReLU()(BatchNormalization()(Conv2D(channel, 3, padding='same', kernel_initializer='he_normal')(conv2)))
    xs.append(conv2)

    conv_unite = Add()(xs)

    avg_pool = GlobalAveragePooling2D(keepdims='True')(conv_unite)
    #output_shape=[b, 1, 1, channel]

    z = ReLU()(BatchNormalization()(Conv2D(channel//ratio, 1, kernel_initializer='he_normal', padding='same')(avg_pool)))
    
    x = Conv2D(channel*2, 1, kernel_initializer='he_normal', padding='same')(z)
    
    x = Reshape([1, 1, channel, 2])(x)

    scale = Softmax()(x)
    
    x = Lambda(lambda x: tf.stack(x, axis=-1),
               output_shape=[b, h, w, channel, 2])(xs)

    f = tf.multiply(scale, x, name='product')
    f = tf.reduce_sum(f, axis=-1, name='sum')
    
    return f[0:4]


def Unet(input_size1 = (192,192,1), input_size2 = (192, 192, 3, 1), num_class=2, n_filt=32):

  input_model1 = Input(input_size1)
  input_model2 = Input(input_size2)

  #layer1 2D
  x1 = ReLU()(BatchNormalization()(Conv2D(n_filt, 3, padding = 'same', kernel_initializer = 'he_normal')(input_model1)))
  conv1 = ReLU()(BatchNormalization()(Conv2D(n_filt, 3, padding = 'same', kernel_initializer = 'he_normal')(x1)))
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  #layer1 3D
  x1_2 = ReLU()(BatchNormalization()(Conv3D(n_filt, 3, padding = 'same', kernel_initializer = 'he_normal')(input_model2)))
  conv1_2 = ReLU()(BatchNormalization()(Conv3D(n_filt, 3, padding = 'same', kernel_initializer = 'he_normal')(x1_2)))
  pool1_2 = MaxPooling3D(pool_size=(2,2,1))(conv1_2)
  #layer2 2D
  conv2 = ReLU()(BatchNormalization()(Conv2D(n_filt*2, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)))
  conv2 = ReLU()(BatchNormalization()(Conv2D(n_filt*2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)))
  #layer2 3D
  conv2_2 = ReLU()(BatchNormalization()(Conv3D(n_filt*2, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1_2)))
  conv2_2 = ReLU()(BatchNormalization()(Conv3D(n_filt*2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2_2)))
  pool2_2 = MaxPooling3D(pool_size=(2,2,1))(conv2_2)

  select1 = selective_kernel(conv2, conv2_2, n_filt*2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(select1)

  #layer3 2D
  conv3 = ReLU()(BatchNormalization()(Conv2D(n_filt*4, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)))
  conv3 = ReLU()(BatchNormalization()(Conv2D(n_filt*4, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)))
  #layer3 3D
  conv3_2 = ReLU()(BatchNormalization()(Conv3D(n_filt*4, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2_2)))
  conv3_2 = ReLU()(BatchNormalization()(Conv3D(n_filt*4, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3_2)))

  select2 = selective_kernel(conv3, conv3_2, n_filt*4)
  pool3 = MaxPooling2D(pool_size=(2, 2))(select2)

  #layer4 2D
  conv4 = ReLU()(BatchNormalization()(Conv2D(n_filt*8, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)))
  conv4 = ReLU()(BatchNormalization()(Conv2D(n_filt*8, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)))
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
  #layer5 2D
  conv5 = ReLU()(BatchNormalization()(Conv2D(n_filt*16, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)))
  conv5 = ReLU()(BatchNormalization()(Conv2D(n_filt*16, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5)))

  conv_up5 = ReLU()(BatchNormalization()(Conv2DTranspose(num_class, 4, strides=(2, 2), padding='same',activation = 'relu',kernel_initializer = 'he_normal')(conv5)))

  merge6 = concatenate([conv_up5,conv4], axis = 3)
  conv6 = ReLU()(BatchNormalization()(Conv2D(n_filt*8, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6)))
  conv6 = ReLU()(BatchNormalization()(Conv2D(n_filt*8, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)))

  conv_up6 = ReLU()(BatchNormalization()(Conv2DTranspose(num_class, 4, strides=(2, 2), padding='same',activation = 'relu',kernel_initializer = 'he_normal')(conv6)))

  merge7 = concatenate([conv_up6,conv3], axis = 3)
  conv7 = ReLU()(BatchNormalization()(Conv2D(n_filt*4, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)))
  conv7 = ReLU()(BatchNormalization()(Conv2D(n_filt*4, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)))

  conv_up7 = ReLU()(BatchNormalization()(Conv2DTranspose(num_class, 4, strides=(2, 2), padding='same',activation = 'relu',kernel_initializer = 'he_normal')(conv7)))

  merge8 = concatenate([conv_up7,conv2], axis = 3)
  conv8 = ReLU()(BatchNormalization()(Conv2D(n_filt*2, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)))
  conv8 = ReLU()(BatchNormalization()(Conv2D(n_filt*2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)))

  conv_up8 = ReLU()(BatchNormalization()(Conv2DTranspose(num_class, 4, strides=(2, 2), padding='same',activation = 'relu',kernel_initializer = 'he_normal')(conv8)))

  merge9 = concatenate([conv_up8,conv1], axis = 3)
  conv9 = ReLU()(BatchNormalization()(Conv2D(n_filt, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)))
  conv9 = ReLU()(BatchNormalization()(Conv2D(n_filt, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)))

  if num_class>2:
      conv_out=Conv2D(num_class, 1, activation = 'softmax', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  else:
      conv_out=Conv2D(num_class-1, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  
  model=Model(inputs=[input_model1, input_model2],outputs=conv_out)
  return model


model = Unet()
model.summary(line_length=140)
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
