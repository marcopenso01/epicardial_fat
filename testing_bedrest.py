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


def compute_metrics_on_slice(img, px_size):
    pred_binary = (img == 1) * 1
    areapred = pred_binary.sum() * (float(px_size[0]) * float(px_size[1])) * float(px_size[2]) / 1000.
    return areapred


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PATH
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
log_root = 'D:\GRASSO\logdir2'
experiment_name = 'ConvMixUnet2_5D_3'
forceoverwrite = True

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TESTING AND EVALUATING THE MODEL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('-' * 50)
print('Testing bed rest...')
print('-' * 50)
model_path = os.path.join(log_root, experiment_name)
test_path = 'G:\BED-REST\patientsESA20221202\patientsESA20221202\pre_process'

for paz in os.listdir(test_path):
    fold_paz = os.path.join(test_path, paz)

    data = h5py.File(os.path.join(fold_paz, 'data.hdf5'), 'r')

    test_img = []
    test_up = []
    test_down = []
    test_paz = []
    test_grasso = []
    test_pred = []
    test_phase = []
    test_slice = []
    px_dim = data['pixel_size'][0]  # px, py, pz

    for k in data.keys():
        if k == 'pixel_size':
            continue
        n_phase = int(k.split('phase')[-1])
        for ii in range(len(data[k])):
            if ii > 1 and ii < len(data[k]) - 1:
                # rimuovo prime e ultime fette
                test_img.append(data[k][ii].astype('float32'))
                test_up.append(data[k][ii - 1].astype('float32'))
                test_down.append(data[k][ii + 1].astype('float32'))
                test_phase.append(n_phase)
                test_paz.append(paz)
                test_slice.append(ii)

    data.close()

    print('Loading saved weights...')
    model = tf.keras.models.load_model(os.path.join(model_path, 'model_weights.h5'),
                                       custom_objects={'loss_function': losses.focal_tversky_loss(),
                                                       'dice_coef': losses.dice_coef})

    logging.info(' --------------------------------------------')
    logging.info('------- Analysing paz: %s' % paz)
    logging.info(' --------------------------------------------')

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
        test_grasso.append(compute_metrics_on_slice(mask_out, px_dim))

    # save data in excel
    print('saving excel file...')
    df = pd.DataFrame({'paz': test_paz, 'phase': test_phase,
                       'vol_fat': test_grasso, 'n_slice': test_slice})
    df.to_excel(os.path.join(test_path, paz, 'excel_df_slice.xlsx'))

    # save data
    print('saving pred...')
    hdf5_file = h5py.File(os.path.join(test_path, paz, 'output.hdf5'), "w")
    dt = h5py.special_dtype(vlen=str)
    hdf5_file.create_dataset('paz', (len(test_paz),), dtype=dt)
    hdf5_file.create_dataset('phase', (len(test_phase),), dtype=dt)
    hdf5_file.create_dataset('n_slice', (len(test_slice),), dtype=np.int)
    hdf5_file.create_dataset('pixel_size', (1, 3), dtype=dt)
    hdf5_file.create_dataset('img_raw', [len(test_img)] + [img.shape[0], img.shape[1]], dtype=np.float32)
    hdf5_file.create_dataset('pred', [len(test_pred)] + [img.shape[0], img.shape[1]], dtype=np.float32)

    for i in range(len(test_img)):
        hdf5_file['paz'][i, ...] = test_paz[i]
        hdf5_file['phase'][i, ...] = test_phase[i]
        hdf5_file['img_raw'][i, ...] = test_img[i]
        hdf5_file['pred'][i, ...] = test_pred[i]
        hdf5_file['n_slice'][i, ...] = test_slice[i]
    hdf5_file['pixel_size'][...] = px_dim
    # After loop:
    hdf5_file.close()

    # save img pred
    print('saving images...')
    pdf_path = os.path.join(test_path, paz, 'plt_imgs.pdf')
    figs = []
    for i in range(len(test_img)):
        img_raw = cv2.normalize(src=test_img[i], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                dtype=cv2.CV_8U)
        pred = test_pred[i].astype(np.uint8)
        fig = plt.figure(figsize=(14, 14))
        ax1 = fig.add_subplot(121)
        ax1.set_axis_off()
        ax1.imshow(img_raw, cmap='gray')

        ax2 = fig.add_subplot(122)
        ax2.set_axis_off()
        ax2.imshow(
            color.label2rgb(pred, img_raw, colors=[(255, 0, 0), (0, 255, 0), (255, 255, 0)], alpha=0.0008, bg_label=0,
                            bg_color=None))
        ax1.title.set_text('Raw_img')
        ax2.title.set_text('Automated')
        txt = str('paz: ' + test_paz[i] + ' phase: ' + str(test_phase[i]) + ' slice: ' + str(test_slice[i]))
        plt.text(0.1, 0.80, txt, transform=fig.transFigure, size=18)
        figs.append(fig)
        # plt.show()

    with PdfPages(pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig)
            plt.close()
