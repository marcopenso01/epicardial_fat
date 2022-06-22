import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import logging

logging.basicConfig(
    level=logging.INFO  # allow DEBUG level messages to pass through the logger
)


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
    # output_shape=[b, 1, 1, channel]

    z = ReLU()(
        BatchNormalization()(Conv2D(channel // ratio, 1, kernel_initializer='he_normal', padding='same')(avg_pool)))

    x = Conv2D(channel * 2, 1, kernel_initializer='he_normal', padding='same')(z)

    x = Reshape([1, 1, channel, 2])(x)

    scale = Softmax()(x)

    x = Lambda(lambda x: tf.stack(x, axis=-1),
               output_shape=[b, h, w, channel, 2])(xs)

    f = tf.multiply(scale, x, name='product')
    f = tf.reduce_sum(f, axis=-1, name='sum')

    return f[0:4]

def selective(input1, input2):
    '''
    input1: input tensor from the 2D network (x,y,channels)
    input2: input tensor from the 3D network (x,y,depth,channels)
    channel: channel number of the result
    return: processed tensor
    '''
    channels = input1.shape[-1]
    conv2 = BatchNormalization()(ReLU()(Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input2)))
    conv2 = tf.keras.backend.squeeze(conv2, axis=-1)
    conv2 = BatchNormalization()(ReLU()(Conv2D(channels, 3, padding='same', kernel_initializer='he_normal')(conv2)))
    return add([input1, conv2])

def Unet(input_size1=(160, 160, 1), n_filt=32):
    input_model1 = Input(input_size1)

    # layer1 2D
    x1 = ReLU()(BatchNormalization()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(input_model1)))
    conv1 = ReLU()(BatchNormalization()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(x1)))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # layer2 2D
    conv2 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, (3, 3), padding='same', kernel_initializer='he_normal')(pool1)))
    conv2 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, (3, 3), padding='same', kernel_initializer='he_normal')(conv2)))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # layer3 2D
    conv3 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, (3, 3), padding='same', kernel_initializer='he_normal')(pool2)))
    conv3 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, (3, 3), padding='same', kernel_initializer='he_normal')(conv3)))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # layer4 2D
    conv4 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, (3, 3), padding='same', kernel_initializer='he_normal')(pool3)))
    conv4 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, (3, 3), padding='same', kernel_initializer='he_normal')(conv4)))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # layer5 2D
    conv5 = ReLU()(BatchNormalization()(Conv2D(n_filt * 16, (3, 3), padding='same', kernel_initializer='he_normal')(pool4)))
    conv5 = ReLU()(BatchNormalization()(Conv2D(n_filt * 16, (3, 3), padding='same', kernel_initializer='he_normal')(conv5)))

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)

    conv6 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, (3, 3), padding='same', kernel_initializer='he_normal')(up4)))
    conv6 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, (3, 3), padding='same', kernel_initializer='he_normal')(conv6)))

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)

    conv7 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, (3, 3), padding='same', kernel_initializer='he_normal')(up3)))
    conv7 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, (3, 3), padding='same', kernel_initializer='he_normal')(conv7)))

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)

    conv8 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, (3, 3), padding='same', kernel_initializer='he_normal')(up2)))
    conv8 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, (3, 3), padding='same', kernel_initializer='he_normal')(conv8)))

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)

    conv9 = ReLU()(BatchNormalization()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(up1)))
    conv9 = ReLU()(BatchNormalization()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(conv9)))

    output = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=input_model1, outputs=output)
    logging.info('Finish building model')

    return model


def custom_gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

def activation_block(x):
    #x = custom_gelu(x)
    x = ReLU()(x)
    return BatchNormalization()(x)

def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = Add()([activation_block(x), x0])  # Residual.
    # Pointwise convolution.
    x = Conv2D(filters, kernel_size=1, kernel_initializer='he_normal')(x)
    x = activation_block(x)
    return x

def convmix_inception_layer(layer_in, f):
    x1 = conv_mixer_block(layer_in, filters=f, kernel_size=3)

    x2 = conv_mixer_block(layer_in, filters=f, kernel_size=3)
    x2 = conv_mixer_block(x2, filters=f, kernel_size=3)

    #x3 = conv_mixer_block(layer_in, filters=f, kernel_size=3)
    #x3 = conv_mixer_block(x3, filters=f, kernel_size=3)
    #x3 = conv_mixer_block(x3, filters=f, kernel_size=3)

    return Concatenate(axis=-1)([x1, x2])


def IncepConvMixUnet(input_size1=(160, 160, 1), n_filt=32):
    input_model1 = Input(input_size1)

    # layer1 2D
    conv1 = ReLU()(BatchNormalization()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(input_model1)))
    conv1 = convmix_inception_layer(conv1, n_filt)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # layer2 2D
    conv2 = convmix_inception_layer(pool1, n_filt * 2)
    conv2 = convmix_inception_layer(conv2, n_filt * 2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # layer3 2D
    conv3 = convmix_inception_layer(pool2, n_filt * 4)
    conv3 = convmix_inception_layer(conv3, n_filt * 4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # layer4 2D
    conv4 = convmix_inception_layer(pool3, n_filt * 8)
    conv4 = convmix_inception_layer(conv4, n_filt * 8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # layer5 2D
    conv5 = convmix_inception_layer(pool4, n_filt * 16)
    conv5 = convmix_inception_layer(conv5, n_filt * 16)

    up4 = UpSampling2D(size=(2, 2))(conv5)
    skip4_0 = conv_mixer_block(conv4, filters=n_filt, kernel_size=3)
    skip4_1 = concatenate([conv4, skip4_0], axis=3)
    conc4 = concatenate([up4, skip4_1], axis=3)

    conv6 = convmix_inception_layer(conc4, n_filt * 8)
    conv6 = convmix_inception_layer(conv6, n_filt * 8)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    skip3_0 = conv_mixer_block(conv3, filters=n_filt, kernel_size=3)
    skip3_1 = concatenate([conv3, skip3_0], axis=3)
    skip3_2 = conv_mixer_block(skip3_1, filters=n_filt, kernel_size=3)
    skip3_3 = concatenate([skip3_1, skip3_2], axis=3)
    conc3 = concatenate([up3, skip3_3], axis=3)

    conv7 = convmix_inception_layer(conc3, n_filt * 4)
    conv7 = convmix_inception_layer(conv7, n_filt * 4)

    up2 = UpSampling2D(size=(2, 2))(conv7)
    skip2_0 = conv_mixer_block(conv2, filters=n_filt, kernel_size=3)
    skip2_1 = concatenate([conv2, skip2_0], axis=3)
    skip2_2 = conv_mixer_block(skip2_1, filters=n_filt, kernel_size=3)
    skip2_3 = concatenate([skip2_1, skip2_2], axis=3)
    skip2_4 = conv_mixer_block(skip2_3, filters=n_filt, kernel_size=3)
    skip2_5 = concatenate([skip2_3, skip2_4], axis=3)
    conc2 = concatenate([up2, skip2_5], axis=3)

    conv8 = convmix_inception_layer(conc2, n_filt * 2)
    conv8 = convmix_inception_layer(conv8, n_filt * 2)

    up1 = UpSampling2D(size=(2, 2))(conv8)
    skip1_0 = conv_mixer_block(conv1, filters=n_filt, kernel_size=3)
    skip1_1 = concatenate([conv1, skip1_0], axis=3)
    skip1_2 = conv_mixer_block(skip1_1, filters=n_filt, kernel_size=3)
    skip1_3 = concatenate([skip1_1, skip1_2], axis=3)
    skip1_4 = conv_mixer_block(skip1_3, filters=n_filt, kernel_size=3)
    skip1_5 = concatenate([skip1_3, skip1_4], axis=3)
    skip1_6 = conv_mixer_block(skip1_5, filters=n_filt, kernel_size=3)
    skip1_7 = concatenate([skip1_5, skip1_6], axis=3)
    conc1 = concatenate([up1, skip1_7], axis=3)

    conv9 = convmix_inception_layer(conc1, n_filt)
    conv9 = convmix_inception_layer(conv9, n_filt)

    output = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=input_model1, outputs=output)
    logging.info('Finish building model')

    return model


def ConvMixUnet(input_size1=(160, 160, 1), n_filt=32):
    input_model1 = Input(input_size1)

    # layer1 2D
    conv1 = naive_inception_module(input_model1, n_filt)
    conv1 = conv_mixer_block(conv1, filters=n_filt, kernel_size=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # layer2 2D
    conv2 = conv_mixer_block(pool1, filters=n_filt * 2, kernel_size=3)
    conv2 = conv_mixer_block(conv2, filters=n_filt * 2, kernel_size=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # layer3 2D
    conv3 = conv_mixer_block(pool2, filters=n_filt * 4, kernel_size=3)
    conv3 = conv_mixer_block(conv3, filters=n_filt * 4, kernel_size=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # layer4 2D
    conv4 = conv_mixer_block(pool3, filters=n_filt * 8, kernel_size=3)
    conv4 = conv_mixer_block(conv4, filters=n_filt * 8, kernel_size=3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # layer5 2D
    conv5 = conv_mixer_block(pool4, filters=n_filt * 16, kernel_size=3)
    conv5 = conv_mixer_block(conv5, filters=n_filt * 16, kernel_size=3)

    up4 = UpSampling2D(size=(2, 2))(conv5)
    skip4_0 = conv_mixer_block(conv4, filters=n_filt, kernel_size=3)
    skip4_1 = concatenate([conv4, skip4_0], axis=3)
    conc4 = concatenate([up4, skip4_1], axis=3)

    conv6 = conv_mixer_block(conc4, filters=n_filt * 8, kernel_size=3)
    conv6 = conv_mixer_block(conv6, filters=n_filt * 8, kernel_size=3)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    skip3_0 = conv_mixer_block(conv3, filters=n_filt, kernel_size=3)
    skip3_1 = concatenate([conv3, skip3_0], axis=3)
    skip3_2 = conv_mixer_block(skip3_1, filters=n_filt, kernel_size=3)
    skip3_3 = concatenate([skip3_1, skip3_2], axis=3)
    conc3 = concatenate([up3, skip3_3], axis=3)

    conv7 = conv_mixer_block(conc3, filters=n_filt * 4, kernel_size=3)
    conv7 = conv_mixer_block(conv7, filters=n_filt * 4, kernel_size=3)

    up2 = UpSampling2D(size=(2, 2))(conv7)
    skip2_0 = conv_mixer_block(conv2, filters=n_filt, kernel_size=3)
    skip2_1 = concatenate([conv2, skip2_0], axis=3)
    skip2_2 = conv_mixer_block(skip2_1, filters=n_filt, kernel_size=3)
    skip2_3 = concatenate([skip2_1, skip2_2], axis=3)
    skip2_4 = conv_mixer_block(skip2_3, filters=n_filt, kernel_size=3)
    skip2_5 = concatenate([skip2_3, skip2_4], axis=3)
    conc2 = concatenate([up2, skip2_5], axis=3)

    conv8 = conv_mixer_block(conc2, filters=n_filt * 2, kernel_size=3)
    conv8 = conv_mixer_block(conv8, filters=n_filt * 2, kernel_size=3)

    up1 = UpSampling2D(size=(2, 2))(conv8)
    skip1_0 = conv_mixer_block(conv1, filters=n_filt, kernel_size=3)
    skip1_1 = concatenate([conv1, skip1_0], axis=3)
    skip1_2 = conv_mixer_block(skip1_1, filters=n_filt, kernel_size=3)
    skip1_3 = concatenate([skip1_1, skip1_2], axis=3)
    skip1_4 = conv_mixer_block(skip1_3, filters=n_filt, kernel_size=3)
    skip1_5 = concatenate([skip1_3, skip1_4], axis=3)
    skip1_6 = conv_mixer_block(skip1_5, filters=n_filt, kernel_size=3)
    skip1_7 = concatenate([skip1_5, skip1_6], axis=3)
    conc1 = concatenate([up1, skip1_7], axis=3)

    conv9 = conv_mixer_block(conc1, filters=n_filt, kernel_size=3)
    conv9 = conv_mixer_block(conv9, filters=n_filt, kernel_size=3)

    output = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=input_model1, outputs=output)
    logging.info('Finish building model')

    return model


def naive_inception_module(layer_in, n_filt=32):
    conv1 = BatchNormalization()(ReLU()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(layer_in)))

    conv2 = BatchNormalization()(ReLU()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(layer_in)))
    conv2 = BatchNormalization()(ReLU()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(conv2)))

    conv3 = BatchNormalization()(
        ReLU()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(layer_in)))
    conv3 = BatchNormalization()(ReLU()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(conv3)))
    conv3 = BatchNormalization()(ReLU()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(conv3)))

    # concatenate filters, assumes filters/channels last
    layer_out = Concatenate(axis=-1)([conv1, conv2, conv3])
    return layer_out


def Unet2_5D(input_size1=(160, 160, 1), input_size2=(160, 160, 1),
          input_size3=(160, 160, 1), num_class=2, n_filt=32):
    input_model1 = Input(input_size1)
    input_model2 = Input(input_size2)
    input_model3 = Input(input_size3)

    # layer1 2D
    x1 = ReLU()(BatchNormalization()(Conv2D(n_filt, 3, padding='same', kernel_initializer='he_normal')(input_model1)))
    conv1 = ReLU()(BatchNormalization()(Conv2D(n_filt, 3, padding='same', kernel_initializer='he_normal')(x1)))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # layer1 3D
    input_model3d = Concatenate(axis=-1)([input_model2, input_model1, input_model3])
    input_model3d = tf.expand_dims(input_model3d, -1)
    x1_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt, 3, padding='same', kernel_initializer='he_normal')(input_model3d)))
    conv1_2 = ReLU()(BatchNormalization()(Conv3D(n_filt, 3, padding='same', kernel_initializer='he_normal')(x1_2)))
    pool1_2 = MaxPooling3D(pool_size=(2, 2, 1))(conv1_2)
    # layer2 2D
    conv2 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, 3, padding='same', kernel_initializer='he_normal')(pool1)))
    conv2 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, 3, padding='same', kernel_initializer='he_normal')(conv2)))
    # layer2 3D
    conv2_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt * 2, 3, padding='same', kernel_initializer='he_normal')(pool1_2)))
    conv2_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt * 2, 3, padding='same', kernel_initializer='he_normal')(conv2_2)))
    pool2_2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2_2)

    select1 = selective_kernel(conv2, conv2_2, n_filt * 2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(select1)

    # layer3 2D
    conv3 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, 3, padding='same', kernel_initializer='he_normal')(pool2)))
    conv3 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, 3, padding='same', kernel_initializer='he_normal')(conv3)))
    # layer3 3D
    conv3_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt * 4, 3, padding='same', kernel_initializer='he_normal')(pool2_2)))
    conv3_2 = ReLU()(
        BatchNormalization()(Conv3D(n_filt * 4, 3, padding='same', kernel_initializer='he_normal')(conv3_2)))

    select2 = selective_kernel(conv3, conv3_2, n_filt * 4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(select2)

    # layer4 2D
    conv4 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, 3, padding='same', kernel_initializer='he_normal')(pool3)))
    conv4 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, 3, padding='same', kernel_initializer='he_normal')(conv4)))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # layer5 2D
    conv5 = ReLU()(BatchNormalization()(Conv2D(n_filt * 16, 3, padding='same', kernel_initializer='he_normal')(pool4)))
    conv5 = ReLU()(BatchNormalization()(Conv2D(n_filt * 16, 3, padding='same', kernel_initializer='he_normal')(conv5)))

    conv_up5 = ReLU()(BatchNormalization()(
        Conv2DTranspose(num_class, 4, strides=(2, 2), padding='same', activation='relu',
                        kernel_initializer='he_normal')(conv5)))

    merge6 = concatenate([conv_up5, conv4], axis=3)
    conv6 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, 3, padding='same', kernel_initializer='he_normal')(merge6)))
    conv6 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, 3, padding='same', kernel_initializer='he_normal')(conv6)))

    conv_up6 = ReLU()(BatchNormalization()(
        Conv2DTranspose(num_class, 4, strides=(2, 2), padding='same', activation='relu',
                        kernel_initializer='he_normal')(conv6)))

    merge7 = concatenate([conv_up6, conv3], axis=3)
    conv7 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, 3, padding='same', kernel_initializer='he_normal')(merge7)))
    conv7 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, 3, padding='same', kernel_initializer='he_normal')(conv7)))

    conv_up7 = ReLU()(BatchNormalization()(
        Conv2DTranspose(num_class, 4, strides=(2, 2), padding='same', activation='relu',
                        kernel_initializer='he_normal')(conv7)))

    merge8 = concatenate([conv_up7, conv2], axis=3)
    conv8 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, 3, padding='same', kernel_initializer='he_normal')(merge8)))
    conv8 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, 3, padding='same', kernel_initializer='he_normal')(conv8)))

    conv_up8 = ReLU()(BatchNormalization()(
        Conv2DTranspose(num_class, 4, strides=(2, 2), padding='same', activation='relu',
                        kernel_initializer='he_normal')(conv8)))

    merge9 = concatenate([conv_up8, conv1], axis=3)
    conv9 = ReLU()(BatchNormalization()(Conv2D(n_filt, 3, padding='same', kernel_initializer='he_normal')(merge9)))
    conv9 = ReLU()(BatchNormalization()(Conv2D(n_filt, 3, padding='same', kernel_initializer='he_normal')(conv9)))

    if num_class > 2:
        conv_out = Conv2D(num_class, 1, activation='softmax', padding='same', kernel_initializer='he_normal')(conv9)
    else:
        conv_out = Conv2D(num_class - 1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv9)

    model = Model(inputs=[input_model1, input_model2, input_model3], outputs=conv_out)

    return model

# sostituisce convolution con convmix nel percorso 2D
def ConvMixUnet2_5D(input_size1=(160, 160, 1), input_size2=(160, 160, 1),
                    input_size3=(160, 160, 1), n_filt=32):
    input_model1 = Input(input_size1)
    input_model2 = Input(input_size2)
    input_model3 = Input(input_size3)

    # layer1 2D
    conv1 = naive_inception_module(input_model1, n_filt)
    conv1 = conv_mixer_block(conv1, filters=n_filt, kernel_size=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # layer1 3D
    input_model3d = Concatenate(axis=-1)([input_model2, input_model1, input_model3])
    input_model3d = tf.expand_dims(input_model3d, -1)
    conv1_3d = BatchNormalization()(ReLU()(Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(input_model3d)))
    conv1_3d = BatchNormalization()(ReLU()(Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(conv1_3d)))
    pool1_3d = MaxPooling3D(pool_size=(2, 2, 1))(conv1_3d)
    # layer2 2D
    conv2 = conv_mixer_block(pool1, filters=n_filt * 2, kernel_size=3)
    conv2 = conv_mixer_block(conv2, filters=n_filt * 2, kernel_size=3)
    # layer2 3D
    conv2_3d = BatchNormalization()(
        ReLU()(Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool1_3d)))
    conv2_3d = BatchNormalization()(ReLU()(Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(conv2_3d)))
    pool2_3d = MaxPooling3D(pool_size=(2, 2, 1))(conv2_3d)

    select1 = selective(conv2, conv2_3d)
    pool2 = MaxPooling2D(pool_size=(2, 2))(select1)
    # layer3 2D
    conv3 = conv_mixer_block(pool2, filters=n_filt * 4, kernel_size=3)
    conv3 = conv_mixer_block(conv3, filters=n_filt * 4, kernel_size=3)
    # layer3 3D
    conv3_3d = BatchNormalization()(
        ReLU()(Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(pool2_3d)))
    conv3_3d = BatchNormalization()(
        ReLU()(Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(conv3_3d)))

    select2 = selective(conv3, conv3_3d)
    pool3 = MaxPooling2D(pool_size=(2, 2))(select2)
    # layer4 2D
    conv4 = conv_mixer_block(pool3, filters=n_filt * 8, kernel_size=3)
    conv4 = conv_mixer_block(conv4, filters=n_filt * 8, kernel_size=3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # layer5 2D
    conv5 = conv_mixer_block(pool4, filters=n_filt * 16, kernel_size=3)
    conv5 = conv_mixer_block(conv5, filters=n_filt * 16, kernel_size=3)

    up4 = UpSampling2D(size=(2, 2))(conv5)
    skip4_0 = conv_mixer_block(conv4, filters=n_filt, kernel_size=3)
    skip4_1 = concatenate([conv4, skip4_0], axis=3)
    conc4 = concatenate([up4, skip4_1], axis=3)

    conv6 = conv_mixer_block(conc4, filters=n_filt * 8, kernel_size=3)
    conv6 = conv_mixer_block(conv6, filters=n_filt * 8, kernel_size=3)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    skip3_0 = conv_mixer_block(conv3, filters=n_filt, kernel_size=3)
    skip3_1 = concatenate([conv3, skip3_0], axis=3)
    skip3_2 = conv_mixer_block(skip3_1, filters=n_filt, kernel_size=3)
    skip3_3 = concatenate([skip3_1, skip3_2], axis=3)
    conc3 = concatenate([up3, skip3_3], axis=3)

    conv7 = conv_mixer_block(conc3, filters=n_filt * 4, kernel_size=3)
    conv7 = conv_mixer_block(conv7, filters=n_filt * 4, kernel_size=3)

    up2 = UpSampling2D(size=(2, 2))(conv7)
    skip2_0 = conv_mixer_block(conv2, filters=n_filt, kernel_size=3)
    skip2_1 = concatenate([conv2, skip2_0], axis=3)
    skip2_2 = conv_mixer_block(skip2_1, filters=n_filt, kernel_size=3)
    skip2_3 = concatenate([skip2_1, skip2_2], axis=3)
    skip2_4 = conv_mixer_block(skip2_3, filters=n_filt, kernel_size=3)
    skip2_5 = concatenate([skip2_3, skip2_4], axis=3)
    conc2 = concatenate([up2, skip2_5], axis=3)

    conv8 = conv_mixer_block(conc2, filters=n_filt * 2, kernel_size=3)
    conv8 = conv_mixer_block(conv8, filters=n_filt * 2, kernel_size=3)

    up1 = UpSampling2D(size=(2, 2))(conv8)
    skip1_0 = conv_mixer_block(conv1, filters=n_filt, kernel_size=3)
    skip1_1 = concatenate([conv1, skip1_0], axis=3)
    skip1_2 = conv_mixer_block(skip1_1, filters=n_filt, kernel_size=3)
    skip1_3 = concatenate([skip1_1, skip1_2], axis=3)
    skip1_4 = conv_mixer_block(skip1_3, filters=n_filt, kernel_size=3)
    skip1_5 = concatenate([skip1_3, skip1_4], axis=3)
    skip1_6 = conv_mixer_block(skip1_5, filters=n_filt, kernel_size=3)
    skip1_7 = concatenate([skip1_5, skip1_6], axis=3)
    conc1 = concatenate([up1, skip1_7], axis=3)

    conv9 = conv_mixer_block(conc1, filters=n_filt, kernel_size=3)
    conv9 = conv_mixer_block(conv9, filters=n_filt, kernel_size=3)

    output = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[input_model1, input_model2, input_model3], outputs=output)
    logging.info('Finish building model')

    return model

# sostituisce conv3D con residual_block3D
def ConvMixUnet2_5D_2(input_size1=(160, 160, 1), input_size2=(160, 160, 1),
                    input_size3=(160, 160, 1), n_filt=32):
    input_model1 = Input(input_size1)
    input_model2 = Input(input_size2)
    input_model3 = Input(input_size3)

    # layer1 2D
    conv1 = naive_inception_module(input_model1, n_filt)
    conv1 = conv_mixer_block(conv1, filters=n_filt, kernel_size=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # layer1 3D
    input_model3d = Concatenate(axis=-1)([input_model2, input_model1, input_model3])
    input_model3d = tf.expand_dims(input_model3d, -1)
    conv1_3d = residual_block3D(32, input_model3d)
    pool1_3d = MaxPooling3D(pool_size=(2, 2, 1))(conv1_3d)
    # layer2 2D
    conv2 = conv_mixer_block(pool1, filters=n_filt * 2, kernel_size=3)
    conv2 = conv_mixer_block(conv2, filters=n_filt * 2, kernel_size=3)
    # layer2 3D
    conv2_3d = residual_block3D(64, pool1_3d)
    pool2_3d = MaxPooling3D(pool_size=(2, 2, 1))(conv2_3d)

    select1 = selective(conv2, conv2_3d)
    pool2 = MaxPooling2D(pool_size=(2, 2))(select1)
    # layer3 2D
    conv3 = conv_mixer_block(pool2, filters=n_filt * 4, kernel_size=3)
    conv3 = conv_mixer_block(conv3, filters=n_filt * 4, kernel_size=3)
    # layer3 3D
    conv3_3d = residual_block3D(128, pool2_3d)

    select2 = selective(conv3, conv3_3d)
    pool3 = MaxPooling2D(pool_size=(2, 2))(select2)
    # layer4 2D
    conv4 = conv_mixer_block(pool3, filters=n_filt * 8, kernel_size=3)
    conv4 = conv_mixer_block(conv4, filters=n_filt * 8, kernel_size=3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # layer5 2D
    conv5 = conv_mixer_block(pool4, filters=n_filt * 16, kernel_size=3)
    conv5 = conv_mixer_block(conv5, filters=n_filt * 16, kernel_size=3)

    up4 = UpSampling2D(size=(2, 2))(conv5)
    skip4_0 = conv_mixer_block(conv4, filters=n_filt, kernel_size=3)
    skip4_1 = concatenate([conv4, skip4_0], axis=3)
    conc4 = concatenate([up4, skip4_1], axis=3)

    conv6 = conv_mixer_block(conc4, filters=n_filt * 8, kernel_size=3)
    conv6 = conv_mixer_block(conv6, filters=n_filt * 8, kernel_size=3)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    skip3_0 = conv_mixer_block(conv3, filters=n_filt, kernel_size=3)
    skip3_1 = concatenate([conv3, skip3_0], axis=3)
    skip3_2 = conv_mixer_block(skip3_1, filters=n_filt, kernel_size=3)
    skip3_3 = concatenate([skip3_1, skip3_2], axis=3)
    conc3 = concatenate([up3, skip3_3], axis=3)

    conv7 = conv_mixer_block(conc3, filters=n_filt * 4, kernel_size=3)
    conv7 = conv_mixer_block(conv7, filters=n_filt * 4, kernel_size=3)

    up2 = UpSampling2D(size=(2, 2))(conv7)
    skip2_0 = conv_mixer_block(conv2, filters=n_filt, kernel_size=3)
    skip2_1 = concatenate([conv2, skip2_0], axis=3)
    skip2_2 = conv_mixer_block(skip2_1, filters=n_filt, kernel_size=3)
    skip2_3 = concatenate([skip2_1, skip2_2], axis=3)
    skip2_4 = conv_mixer_block(skip2_3, filters=n_filt, kernel_size=3)
    skip2_5 = concatenate([skip2_3, skip2_4], axis=3)
    conc2 = concatenate([up2, skip2_5], axis=3)

    conv8 = conv_mixer_block(conc2, filters=n_filt * 2, kernel_size=3)
    conv8 = conv_mixer_block(conv8, filters=n_filt * 2, kernel_size=3)

    up1 = UpSampling2D(size=(2, 2))(conv8)
    skip1_0 = conv_mixer_block(conv1, filters=n_filt, kernel_size=3)
    skip1_1 = concatenate([conv1, skip1_0], axis=3)
    skip1_2 = conv_mixer_block(skip1_1, filters=n_filt, kernel_size=3)
    skip1_3 = concatenate([skip1_1, skip1_2], axis=3)
    skip1_4 = conv_mixer_block(skip1_3, filters=n_filt, kernel_size=3)
    skip1_5 = concatenate([skip1_3, skip1_4], axis=3)
    skip1_6 = conv_mixer_block(skip1_5, filters=n_filt, kernel_size=3)
    skip1_7 = concatenate([skip1_5, skip1_6], axis=3)
    conc1 = concatenate([up1, skip1_7], axis=3)

    conv9 = conv_mixer_block(conc1, filters=n_filt, kernel_size=3)
    conv9 = conv_mixer_block(conv9, filters=n_filt, kernel_size=3)

    output = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[input_model1, input_model2, input_model3], outputs=output)
    logging.info('Finish building model')

    return model

# modifica skip connection sostituendo dense path con residual path
def ConvMixUnet2_5D_3(input_size1=(160, 160, 1), input_size2=(160, 160, 1),
                    input_size3=(160, 160, 1), n_filt2D=32, n_fild3D=32):
    input_model1 = Input(input_size1)
    input_model2 = Input(input_size2)
    input_model3 = Input(input_size3)

    # layer1 2D
    conv1 = naive_inception_module(input_model1, n_filt2D)
    conv1 = conv_mixer_block(conv1, filters=n_filt2D, kernel_size=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # layer1 3D
    input_model3d = Concatenate(axis=-1)([input_model2, input_model1, input_model3])
    input_model3d = tf.expand_dims(input_model3d, -1)
    conv1_3d = residual_block3D(n_fild3D, input_model3d)
    pool1_3d = MaxPooling3D(pool_size=(2, 2, 1))(conv1_3d)
    # layer2 2D
    conv2 = conv_mixer_block(pool1, filters=n_filt2D * 2, kernel_size=3)
    conv2 = conv_mixer_block(conv2, filters=n_filt2D * 2, kernel_size=3)
    # layer2 3D
    conv2_3d = residual_block3D(n_fild3D*2, pool1_3d)
    pool2_3d = MaxPooling3D(pool_size=(2, 2, 1))(conv2_3d)

    select1 = selective(conv2, conv2_3d)
    pool2 = MaxPooling2D(pool_size=(2, 2))(select1)
    # layer3 2D
    conv3 = conv_mixer_block(pool2, filters=n_filt2D * 4, kernel_size=3)
    conv3 = conv_mixer_block(conv3, filters=n_filt2D * 4, kernel_size=3)
    # layer3 3D
    conv3_3d = residual_block3D(n_fild3D*4, pool2_3d)

    select2 = selective(conv3, conv3_3d)
    pool3 = MaxPooling2D(pool_size=(2, 2))(select2)
    # layer4 2D
    conv4 = conv_mixer_block(pool3, filters=n_filt2D * 8, kernel_size=3)
    conv4 = conv_mixer_block(conv4, filters=n_filt2D * 8, kernel_size=3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # layer5 2D
    conv5 = conv_mixer_block(pool4, filters=n_filt2D * 16, kernel_size=3)
    conv5 = conv_mixer_block(conv5, filters=n_filt2D * 16, kernel_size=3)

    up4 = UpSampling2D(size=(2, 2))(conv5)
    up4 = conv2d_bn(up4, n_filt2D * 8, 1, 1, activation=None, padding='same')
    skip4 = ResPath(conv4, length=1)
    conc4 = add([up4, skip4])
    conc4 = Activation('relu')(conc4)
    conc4 = BatchNormalization(axis=3)(conc4)

    conv6 = conv_mixer_block(conc4, filters=n_filt2D * 8, kernel_size=3)
    conv6 = conv_mixer_block(conv6, filters=n_filt2D * 8, kernel_size=3)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    up3 = conv2d_bn(up3, n_filt2D * 4, 1, 1, activation=None, padding='same')
    skip3 = ResPath(conv3, length=2)
    conc3 = add([up3, skip3])
    conc3 = Activation('relu')(conc3)
    conc3 = BatchNormalization(axis=3)(conc3)

    conv7 = conv_mixer_block(conc3, filters=n_filt2D * 4, kernel_size=3)
    conv7 = conv_mixer_block(conv7, filters=n_filt2D * 4, kernel_size=3)

    up2 = UpSampling2D(size=(2, 2))(conv7)
    up2 = conv2d_bn(up2, n_filt2D * 2, 1, 1, activation=None, padding='same')
    skip2 = ResPath(conv2, length=3)
    conc2 = add([up2, skip2])
    conc2 = Activation('relu')(conc2)
    conc2 = BatchNormalization(axis=3)(conc2)

    conv8 = conv_mixer_block(conc2, filters=n_filt2D * 2, kernel_size=3)
    conv8 = conv_mixer_block(conv8, filters=n_filt2D * 2, kernel_size=3)

    up1 = UpSampling2D(size=(2, 2))(conv8)
    up1 = conv2d_bn(up1, n_filt2D, 1, 1, activation=None, padding='same')
    skip1 = ResPath(conv1, length=4)
    conc1 = add([up1, skip1])
    conc1 = Activation('relu')(conc1)
    conc1 = BatchNormalization(axis=3)(conc1)

    conv9 = conv_mixer_block(conc1, filters=n_filt2D, kernel_size=3)
    conv9 = conv_mixer_block(conv9, filters=n_filt2D, kernel_size=3)

    output = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[input_model1, input_model2, input_model3], outputs=output)
    logging.info('Finish building model')

    return model


def ResPath(encoder, length=1):
    '''
    ResPath
    Arguments:
        length {int} -- length of ResPath
        encoder {keras layer} -- input encoder layer
        decoder {keras layer} -- input decoder layer
    Returns:
        [keras layer] -- [output layer]
    '''
    channels = encoder.shape[-1]
    shortcut = encoder
    shortcut = conv2d_bn(shortcut, channels, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(encoder, channels, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length - 1):
        shortcut = out
        shortcut = conv2d_bn(shortcut, channels, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, channels, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


def MultiResBlock(U, inp, alpha=1.67):
    '''
    MultiRes Block

    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W * 0.167) + int(W * 0.333) +
                         int(W * 0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W * 0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W * 0.333), 3, 3,
                        activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W * 0.5), 3, 3,
                        activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    return x


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if (activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def MultiResUnet(input_size1=(160, 160, 1)):
    inputs = Input(input_size1)

    mresblock1 = MultiResBlock(48, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(48, 4, mresblock1)

    mresblock2 = MultiResBlock(48 * 2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(48 * 2, 3, mresblock2)

    mresblock3 = MultiResBlock(48 * 4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(48 * 4, 2, mresblock3)

    mresblock4 = MultiResBlock(48 * 8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(48 * 8, 1, mresblock4)

    mresblock5 = MultiResBlock(48 * 16, pool4)

    up6 = concatenate([Conv2DTranspose(
        48 * 8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(48 * 8, up6)

    up7 = concatenate([Conv2DTranspose(
        48 * 4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(48 * 4, up7)

    up8 = concatenate([Conv2DTranspose(
        48 * 2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(48 * 2, up8)

    up9 = concatenate([Conv2DTranspose(48, (2, 2), strides=(
        2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(48, up9)

    conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')

    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def residual_block(filters, inp):
    x = Conv3D(filters, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=True)(inp)
    x = BatchNormalization(axis=3, scale=False)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=True)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    x = Activation('relu')(x)
    shortcut = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=True)(
        inp)
    shortcut = BatchNormalization(axis=3, scale=False)(shortcut)
    output = add([shortcut, x])

    return output

def residual_block3D(filters, inp):
    x = BatchNormalization()(
        ReLU()(Conv3D(filters, 3, padding='same', kernel_initializer='he_normal')(inp)))

    x = BatchNormalization()(
        ReLU()(Conv3D(filters, 3, padding='same', kernel_initializer='he_normal')(x)))

    shortcut = Conv3D(filters, 1, padding='same', kernel_initializer='he_normal')(inp)
    output = add([shortcut, x])
    output = Activation('relu')(output)
    output = BatchNormalization(axis=3, scale=False)(output)

    return output


def ResUnet(input_size1=(160, 160, 1)):
    n_filt = 32
    inputs = Input(input_size1)

    e1 = residual_block(n_filt, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(e1)
    e2 = residual_block(n_filt * 2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(e2)
    e3 = residual_block(n_filt * 4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(e3)
    e4 = residual_block(n_filt * 8, pool3)
    d1 = Dropout(0.5)(e4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(d1)
    e5 = residual_block(n_filt * 16, pool4)
    d2 = Dropout(0.5)(e5)

    up4 = UpSampling2D(size=(2, 2))(d2)
    conc4 = concatenate([up4, e4], axis=3)
    d4 = residual_block(n_filt * 8, conc4)

    up3 = UpSampling2D(size=(2, 2))(d4)
    conc3 = concatenate([up3, e3], axis=3)
    d3 = residual_block(n_filt * 4, conc3)

    up2 = UpSampling2D(size=(2, 2))(d3)
    conc2 = concatenate([up2, e2], axis=3)
    d2 = residual_block(n_filt * 2, conc2)

    up1 = UpSampling2D(size=(2, 2))(d2)
    conc1 = concatenate([up1, e1], axis=3)
    d1 = residual_block(n_filt, conc1)

    pred = Conv2D(1, (1, 1), activation='sigmoid')(d1)

    model = Model(inputs=[inputs], outputs=[pred])

    return model
