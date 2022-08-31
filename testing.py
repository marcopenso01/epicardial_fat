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


def myprint(s):
    with open('modelsummary.txt', 'w+') as f:
        print(s, file=f)


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='nearest', cval=0., interpolation_order=1):
    """Performs a random spatial zoom of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        interpolation_order: int, order of spline interpolation.
            see `ndimage.interpolation.affine_transform`
    # Returns
        Zoomed Numpy image tensor.
    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two'
                         ' floats. Received: %s' % (zoom_range,))

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    x = apply_affine_transform(x, zx=zx, zy=zy, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval,
                               order=interpolation_order)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 - 0.5
    o_y = float(y) / 2 - 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_affine_transform(img1, img2, img3, lbl, rows, cols,
                           theta=0, tx=0, ty=0, zx=1, zy=1,
                           fill_mode='nearest', order=1):
    '''
    Applies an affine transformation specified by the parameters given.
    :param img: A numpy array of shape [x, y, nchannels]
    :param rows: img rows
    :param cols: img cols
    :param theta: Rotation angle in degrees
    :param tx: Width shift
    :param ty: Heigh shift
    :param fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
    :param int, order of interpolation
    :return The transformed version of the input
    '''
    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if transform_matrix is not None:
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, rows, cols)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [ndimage.interpolation.affine_transform(
            img1[:, :, channel],
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=0.0) for channel in range(img1.shape[-1])]
        img1 = np.stack(channel_images, axis=2)

        channel_images = [ndimage.interpolation.affine_transform(
            img2[:, :, channel],
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=0.0) for channel in range(img2.shape[-1])]
        img2 = np.stack(channel_images, axis=2)

        channel_images = [ndimage.interpolation.affine_transform(
            img3[:, :, channel],
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=0.0) for channel in range(img3.shape[-1])]
        img3 = np.stack(channel_images, axis=2)

        channel_images = [ndimage.interpolation.affine_transform(
            lbl[:, :, channel],
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=0.0) for channel in range(lbl.shape[-1])]
        lbl = np.stack(channel_images, axis=2)

    return img1, img2, img3, lbl


def augmentation_function(image1, image2, image3, labels):
    '''
    Function for augmentation of minibatches.
    :param images: A numpy array of shape [minibatch, X, Y, nchannels]
    :param labels: A numpy array containing a corresponding label mask
    :return: A mini batch of the same size but with transformed images and masks. 
    '''

    if image1.ndim > 4:
        raise AssertionError('Augmentation will only work with 2D images')

    new_images1 = []
    new_images2 = []
    new_images3 = []
    new_labels = []
    num_images = image1.shape[0]
    rows = image1.shape[1]
    cols = image1.shape[2]

    for ii in range(num_images):

        img1 = image1[ii, ...]
        img2 = image2[ii, ...]
        img3 = image3[ii, ...]
        lbl = labels[ii, ...]

        # ROTATE
        angles = (-30, 30)
        theta = np.random.uniform(angles[0], angles[1])

        # RANDOM WIDTH SHIFT
        width_rg = 0.05
        if width_rg >= 1:
            ty = np.random.choice(int(width_rg))
            ty *= np.random.choice([-1, 1])
        elif width_rg >= 0 and width_rg < 1:
            ty = np.random.uniform(-width_rg,
                                   width_rg)
            ty = int(ty * cols)
        else:
            raise ValueError("do_width_shift_range parameter should be >0")

        # RANDOM HEIGHT SHIFT
        height_rg = 0.05
        if height_rg >= 1:
            tx = np.random.choice(int(height_rg))
            tx *= np.random.choice([-1, 1])
        elif height_rg >= 0 and height_rg < 1:
            tx = np.random.uniform(-height_rg,
                                   height_rg)
            tx = int(tx * rows)
        else:
            raise ValueError("do_height_shift_range parameter should be >0")

        # ZOOM
        # Float or [lower, upper].Range for random zoom.
        # If a float, `[lower, upper] = [1 - zoom_range, 1 + zoom_range]`
        zx, zy = np.random.uniform(1 - 0.05, 1 + 0.05, 2)

        # RANDOM HORIZONTAL FLIP
        flip_horizontal = (np.random.random() < 0.5)

        # RANDOM VERTICAL FLIP
        flip_vertical = (np.random.random() < 0.5)

        img1, img2, img3, lbl = apply_affine_transform(img1, img2, img3, lbl,
                                                       rows=rows, cols=cols,
                                                       theta=theta, tx=tx, ty=ty,
                                                       zx=zx, zy=zy,
                                                       fill_mode='nearest',
                                                       order=1)

        if flip_horizontal:
            img1 = flip_axis(img1, 1)
            img2 = flip_axis(img2, 1)
            img3 = flip_axis(img3, 1)
            lbl = flip_axis(lbl, 1)

        if flip_vertical:
            img1 = flip_axis(img1, 0)
            img2 = flip_axis(img2, 0)
            img3 = flip_axis(img3, 0)
            lbl = flip_axis(lbl, 0)

        new_images1.append(img1)
        new_images2.append(img2)
        new_images3.append(img3)
        new_labels.append(lbl)

    sampled_image1_batch = np.asarray(new_images1)
    sampled_image2_batch = np.asarray(new_images2)
    sampled_image3_batch = np.asarray(new_images3)
    sampled_label_batch = np.asarray(new_labels)

    return sampled_image1_batch, sampled_image2_batch, sampled_image3_batch, sampled_label_batch


def iterate_minibatches(image1, image2, image3, labels, batch_size, augment_batch=False, expand_dims=True):
    '''
    Function to create mini batches from the dataset of a certain batch size 
    :param images: input data shape (N, W, H)
    :param labels: label data
    :param batch_size: batch size (Int)
    :param augment_batch: should batch be augmented?, Boolean (default: False)
    :param expand_dims: adding a dimension, Boolean (default: True)
    :return: mini batches
    '''
    random_indices = np.arange(image1.shape[0])
    np.random.shuffle(random_indices)
    n_images = image1.shape[0]
    for b_i in range(0, n_images, batch_size):

        if b_i + batch_size > n_images:
            continue

        batch_indices = np.sort(random_indices[b_i:b_i + batch_size])
        X1 = image1[batch_indices, ...]  # array of shape [minibatch, X, Y]
        X2 = image2[batch_indices, ...]
        X3 = image3[batch_indices, ...]
        y = labels[batch_indices, ...]

        if expand_dims:
            X1 = X1[..., np.newaxis]  # array of shape [minibatch, X, Y, nchannels=1]
            X2 = X2[..., np.newaxis]
            X3 = X3[..., np.newaxis]
            y = y[..., np.newaxis]

        if augment_batch:
            X1, X2, X3, y = augmentation_function(X1, X2, X3, y)

        yield X1, X2, X3, y


def do_eval(image1, image2, image3, labels, batch_size, augment_batch=False, expand_dims=True):
    '''
    Function for running the evaluations on the validation sets.  
    :param images: A numpy array containing the images
    :param labels: A numpy array containing the corresponding labels 
    :param batch_size: batch size
    :param augment_batch: should batch be augmented?
    :param expand_dims: adding a dimension to a tensor? 
    :return: Scalar val loss and metrics
    '''
    num_batches = 0
    history = []
    for batch in iterate_minibatches(image1,
                                     image2,
                                     image3,
                                     labels,
                                     batch_size=batch_size,
                                     augment_batch=augment_batch,
                                     expand_dims=expand_dims):
        x1, x2, x3, y = batch
        if y.shape[0] < batch_size:
            continue

        val_hist = model.test_on_batch((x1, x2, x3), y)
        if history == []:
            history.append(val_hist)
        else:
            history[0] = [i + j for i, j in zip(history[0], val_hist)]
        num_batches += 1

    for i in range(len(history[0])):
        history[0][i] /= num_batches

    return history[0]


def print_txt(output_dir, stringa):
    out_file = os.path.join(output_dir, 'summary_report.txt')
    with open(out_file, "a") as text_file:
        text_file.writelines(stringa)

def compute_metrics_on_patient(input_fold):
    data = h5py.File(os.path.join(input_fold, 'pred.hdf5'), 'r')
    file_names = []
    # measures per structure:
    vol_list = []
    for paz in np.unique(data['paz'][()]):
        pred_arr = []  # predizione del modello
        flag = 1
        for i in np.where(data['paz'][()] == paz)[0]:
            pred_arr.append(data['pred'][i])
            if flag:
                px_size = data['pixel_size'][i]
                flag = 0

        pred_arr = np.transpose(np.asarray(pred_arr, dtype=np.uint8), (1, 2, 0))

        pred_binary = (pred_arr == 1) * 1
        volpred = pred_binary.sum() * (float(px_size[0]) * float(px_size[1])) * float(px_size[2]) / 1000.
        vol_list.append(volpred)  # volume predetto CNN
        file_names.append(paz)

    df = pd.DataFrame({'vol': vol_list, 'paz': file_names})
    data.close()
    return df


def compute_metrics_on_slice(input_fold):
    data = h5py.File(os.path.join(input_fold, 'pred.hdf5'), 'r')
    file_names = []
    # measures per structure:
    vol_list = []
    for i in range(len(data['pred'][()])):
        pred_binary = (data['pred'][i] == 1) * 1
        px_size = data['pixel_size'][i]

        areapred = pred_binary.sum() * (float(px_size[0]) * float(px_size[1])) * 1 / 1000.
        vol_list.append(areapred)  # volume predetto CNN
        file_names.append(data['paz'][i])

    df = pd.DataFrame({'vol': vol_list, 'paz': file_names})
    data.close()
    return df

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PATH
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
log_root = 'D:\GRASSO\logdir2'
experiment_name = 'ConvMixUnet2_5D_3'
forceoverwrite = True

out_fold = os.path.join(log_root, experiment_name)
out_file = os.path.join(out_fold, 'summary_report.txt')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TESTING AND EVALUATING THE MODEL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('-' * 50)
print('Testing...')
print('-' * 50)
test_path = os.path.join(out_fold, 'predictions_3')

if not tf.io.gfile.exists(test_path):
    tf.io.gfile.makedirs(test_path)
data_file_path = os.path.join(test_path, 'pred.hdf5')
out_pred_data = h5py.File(data_file_path, "w")

data = h5py.File(os.path.join('D:\GRASSO\data', 'test_63.hdf5'), 'r')
test_img = data['img_raw'][()].astype('float32')
test_up = data['img_up'][()].astype('float32')
test_down = data['img_down'][()].astype('float32')
test_paz = data['paz'][()]
test_px = data['pixel_size'][()]
data.close()

with open(out_file, "a") as text_file:
    text_file.write('\n----- test Data summary -----')
print_txt(out_fold, ['\nTesting Images size: %s %s %s' % (test_img.shape[0], test_img.shape[1], test_img.shape[2])])
print_txt(out_fold, ['\nTesting Images type: %s' % test_img.dtype])
print_txt(out_fold, ['\nTesting Up-Images size: %s %s %s' % (test_up.shape[0], test_up.shape[1], test_up.shape[2])])
print_txt(out_fold, ['\nTesting Up-Images type: %s' % test_up.dtype])
print_txt(out_fold,
          ['\nTesting Down-Images size: %s %s %s' % (test_down.shape[0], test_down.shape[1], test_down.shape[2])])
print_txt(out_fold, ['\nTesting Down-Images type: %s' % test_down.dtype])

print('Loading saved weights...')
model = tf.keras.models.load_model(os.path.join(out_fold, 'model_weights.h5'),
                                   custom_objects={'loss_function': losses.focal_tversky_loss(),
                                                   'dice_coef': losses.dice_coef})

RAW = []
PRED = []
PAZ = []
PIXEL = []

total_time = 0
total_volumes = 0

for paz in np.unique(test_paz):
    start_time = time.time()
    logging.info(' --------------------------------------------')
    logging.info('------- Analysing paz: %s' % paz)
    logging.info(' --------------------------------------------')

    for ii in np.where(test_paz == paz)[0]:
        img = test_img[ii]
        img_up = test_up[ii]
        img_down = test_down[ii]
        RAW.append(img)
        PAZ.append(paz)
        PIXEL.append(test_px[ii])

        img = np.float32(normalize_image(img))
        img_up = np.float32(normalize_image(img_up))
        img_down = np.float32(normalize_image(img_down))
        x1 = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
        x2 = np.reshape(img_up, (1, img_up.shape[0], img_up.shape[1], 1))
        x3 = np.reshape(img_down, (1, img_down.shape[0], img_down.shape[1], 1))
        mask_out = model.predict((x1, x2, x3))
        mask_out = np.squeeze(mask_out)
        PRED.append(mask_out)
    elapsed_time = time.time() - start_time
    total_time += elapsed_time
    total_volumes += 1
    logging.info('Evaluation of volume took %f secs.' % elapsed_time)
    print_txt(out_fold, ['\nEvaluation of volume took %f secs.' % elapsed_time])

n_file = len(PRED)
dt = h5py.special_dtype(vlen=str)
out_pred_data.create_dataset('img_raw', [n_file] + [160, 160], dtype=np.float32)
out_pred_data.create_dataset('pred', [n_file] + [160, 160], dtype=np.uint8)
out_pred_data.create_dataset('pixel_size', (n_file, 3), dtype=dt)
out_pred_data.create_dataset('paz', (n_file,), dtype=dt)

for i in range(n_file):
    out_pred_data['img_raw'][i, ...] = RAW[i]
    out_pred_data['pred'][i, ...] = PRED[i]
    out_pred_data['paz'][i, ...] = PAZ[i]
    out_pred_data['pixel_size'][i, ...] = PIXEL[i]

out_pred_data.close()
logging.info('Average time per volume: %f' % (total_time / total_volumes))
print_txt(out_fold, ['\nAverage time per volume: %f' % (total_time / total_volumes)])

print(test_path)
if os.path.exists(os.path.join(test_path, 'pred.hdf5')):
    path_eval = os.path.join(test_path, 'evaluation')
    if not os.path.exists(path_eval):
        tf.io.gfile.makedirs(path_eval)
        print(path_eval)
        df = compute_metrics_on_patient(test_path)
        df.to_pickle(os.path.join(path_eval, 'df_paz.pkl'))
        df.to_excel(os.path.join(path_eval, 'excel_df_paz.xlsx'))

        df = compute_metrics_on_slice(test_path)
        df.to_pickle(os.path.join(path_eval, 'df_slice.pkl'))
        df.to_excel(os.path.join(path_eval, 'excel_df_slice.xlsx'))

print('saving images...')
pdf_path = os.path.join(path_eval, 'plt_imgs.pdf')
data = h5py.File(os.path.join(test_path, 'pred.hdf5'), 'r')
figs = []
for i in range(len(data['img_raw'])):
    img_raw = cv2.normalize(src=data['img_raw'][i], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_8U)
    pred = data['pred'][i].astype(np.uint8)
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
    txt = str(data['paz'][i] + '_' + str(i))
    plt.text(0.1, 0.65, txt, transform=fig.transFigure, size=18)
    figs.append(fig)
    # plt.show()
data.close()

with PdfPages(pdf_path) as pdf:
    for fig in figs:
        pdf.savefig(fig)
        plt.close()
