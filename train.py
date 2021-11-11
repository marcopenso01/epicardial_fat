import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import loss-functions as ls


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


def Unet(input_size1 = (160,160,1), input_size2 = (160, 160, 1), input_size3= (160,160,1), num_class=2, n_filt=32):
  
  input_model1 = Input(input_size1)
  input_model2 = Input(np.stack((input_size2, input_size1, input_size3), axis=-1))

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

"""
GENERATE THE DATA
"""
data_root = 'xxxxx'
model_name = 'xxx'

model = Unet()
model.summary(line_length=140)
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

"""
Load data train
"""
data = h5py.File(os.path.join(data_root, 'train.hdf5'), 'r')

train_img = data['img_raw'][()]
train_label = data['mask'][()]
train_up = data['img_up'][()]
train_down = data['img_down'][()]
train_left = data['img_left'][()]
train_right = data['img_right'][()]
data.close()

data = h5py.File(os.path.join(data_root, 'val.hdf5'), 'r')

val_img = data['img_raw'][()]
val_label = data['mask'][()]
val_up = data['img_up'][()]
val_down = data['img_down'][()]
val_left = data['img_left'][()]
val_right = data['img_right'][()]
data.close()

logging.info('Data summary:')
logging.info(' - Training Images:')
logging.info(train_img.shape)
logging.info(train_img.dtype)
logging.info(' - Training Labels:')
logging.info(train_label.shape)
logging.info(train_label.dtype)
logging.info(' - validation Images:')
logging.info(val_img.shape)
logging.info(val_img.dtype)
logging.info(' - Validation Labels:')
logging.info(val_label.shape)
logging.info(val_label.dtype)

batch_size = 2
num_img = len(train_img)
initial_learning_rate = 1e-3

class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, initial_learning_rate):
    self.initial_learning_rate = initial_learning_rate

  def __call__(self, step):
     return self.initial_learning_rate / (step + 1)

opt = tf.keras.optimizers.Adam(learning_rate=MyLRSchedule(initial_learning_rate))
model.compile(optimizer=opt, loss=ls.unified_focal_loss(weight=0.5, delta=0.6, gamma=0.2), metrics = [ls.dice_coefficient()])

# Define callbacks
checkpoint = ModelCheckpoint(filepath=model_name + '_model_weight.h5',
                             monitor='val_dice_coef',
                             save_best_only=True,
                             save_weights_only=True)

datagen = ImageDataGenerator(
    rotation_range=60,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    interpolation=')

history = model.fit(datagen.flow([train_img, train_up, train_down], train_label, 
                    batch_size = batch_size, subset = 'training'),
                    validation_data=datagen.flow([val_img, val_up, val_down], val_label,
                    batch_size = batch_size, subset = 'validation'),
                    epochs = 200,
                    steps_per_epoch = num_img//batch_size
                    callbacks=[checkpoint],
                    shuffle=True)

print('Model correctly trained and saved')
