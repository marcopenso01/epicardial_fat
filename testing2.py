import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# for GPU process:
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np
import h5py
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
from scipy import ndimage
import shutil
import time
import math
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import logging
import model_structure
import losses
import metrics
import binary_metric as bm
import pandas as pd
import cv2
from skimage import color
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.python.client import device_lib

logging.basicConfig(
    level=logging.INFO  # allow DEBUG level messages to pass through the logger
)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
assert 'GPU' in str(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print('is_gpu_available: %s' % tf.test.is_gpu_available())  # True/False
# Or only check for gpu's with cuda support
print('gpu with cuda support: %s' % tf.test.is_gpu_available(cuda_only=True))
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


# tf.config.list_physical_devices('GPU') #The above function is deprecated in tensorflow > 2.1

def standardize_image(image):
    '''
    make image zero mean and unit standard deviation
    '''

    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s)


def normalize_image(image):
    '''
    make image normalize between 0 and 1
    '''
    img_o = np.float32(image.copy())
    img_o = (img_o - img_o.min()) / (img_o.max() - img_o.min())
    return img_o


def compute_metrics(test_pred, test_img, test_px):
    vol = 0
    max_value = 0
    mean_value = 0
    somma = 0
    count = 0
    n_slice = len(test_pred)
    for i in range(n_slice):
        pred_binary = (test_pred[i] == 1) * 1
        px_size = test_px[i]
        vol = vol + (pred_binary.sum() * (float(px_size[0]) * float(px_size[1])) * float(px_size[2]) / 1000)
        coord = np.where(test_pred[i]==1)
        for ii in range(len(coord[0])):
            somma = somma + test_img[i][coord[0][ii],coord[1][ii]]
            count += 1
            if test_img[i][coord[0][ii],coord[1][ii]] > max_value:
                max_value = test_img[i][coord[0][ii],coord[1][ii]]
    mean_value = int(somma/count)
    return vol, max_value, mean_value

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PATH
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
log_root = 'D:\GRASSO\logdir2'
experiment_name = 'ConvMixUnet2_5D_3'
forceoverwrite = True
n_slice = 5  #5,6,7,0=all   numero di slice partendo dalla basale che si vogliono includere nell'analisi
input_fold = 'G:\DELINEATE CAD\Nuova cartella\group1'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TESTING AND EVALUATING THE MODEL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('-' * 50)
print('Testing...')
print('-' * 50)
model_path = os.path.join(log_root, experiment_name)

logging.info('------- loading model -----------')
model = tf.keras.models.load_model(os.path.join(model_path, 'model_weights.h5'),
                                   custom_objects={'loss_function': losses.focal_tversky_loss(),
                                                   'dice_coef': losses.dice_coef})

VOL = []
MAX = []
PAZ = []
MEAN = []

for paz in os.listdir(input_fold):
    logging.info(' --------------------------------------------')
    logging.info('------- Analysing paz: %s' % paz)
    logging.info(' --------------------------------------------')
   
    data = h5py.File(os.path.join(input_fold, paz, 'pre_processing', 'pre_proc.hdf5'), 'r')
    if n_slice == 0:
        test_img = data['img_raw'][()].astype('float32')
        test_up = data['img_up'][()].astype('float32')
        test_down = data['img_down'][()].astype('float32')
        test_px = data['pixel_size'][()]
    else:
        test_img = data['img_raw'][0:n_slice].astype('float32')
        test_up = data['img_up'][0:n_slice].astype('float32')
        test_down = data['img_down'][0:n_slice].astype('float32')
        test_px = data['pixel_size'][0:n_slice]
    data.close()
    
    test_pred = []

    for ii in range(len(test_img)):
      
        img = test_img[ii]
        img_up = test_up[ii]
        img_down = test_down[ii]

        img = np.float32(normalize_image(img))
        img_up = np.float32(normalize_image(img_up))
        img_down = np.float32(normalize_image(img_down))
        
        x1 = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
        x2 = np.reshape(img_up, (1, img_up.shape[0], img_up.shape[1], 1))
        x3 = np.reshape(img_down, (1, img_down.shape[0], img_down.shape[1], 1))
        
        mask_out = model.predict((x1, x2, x3))
        mask_out = np.squeeze(mask_out)
        test_pred.append(mask_out)
    
    vol, max_value, mean_value = compute_metrics(test_pred, test_img, test_px)
    
    VOL.append(vol)
    MAX.append(max_value)
    MEAN.append(mean_value)
    PAZ.append(paz)
    
df = pd.DataFrame({'paz': PAZ, 'vol': VOL,
                   'max': MAX, 'mean': MEAN})

output_path = 'D:\GRASSO\database'
df.to_excel(os.path.join(output_path, 'excel_measurement.xlsx'))
