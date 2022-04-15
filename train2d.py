import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# for GPU process:
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# for CPU process:
# os.environ["CUDA_VISIBLE_DEVICES"] =

import numpy as np
import h5py
# import skimage.io as io
# import skimage.transform as trans
import matplotlib.pyplot as plt
from scipy import ndimage
import shutil
import time
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import logging
import model_structure
import losses
import metrics
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


def apply_affine_transform(img, lbl, rows, cols, theta=0, tx=0, ty=0,
                           zx=1, zy=1, fill_mode='nearest', order=1):
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

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, rows, cols)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [ndimage.interpolation.affine_transform(
            img[:, :, channel],
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=0.0) for channel in range(img.shape[-1])]
        img = np.stack(channel_images, axis=2)

        channel_images = [ndimage.interpolation.affine_transform(
            lbl[:, :, channel],
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=0.0) for channel in range(lbl.shape[-1])]
        lbl = np.stack(channel_images, axis=2)

    return img, lbl


def augmentation_function(images, labels):
    '''
    Function for augmentation of minibatches.
    :param images: A numpy array of shape [minibatch, X, Y, nchannels]
    :param labels: A numpy array containing a corresponding label mask
    :return: A mini batch of the same size but with transformed images and masks.
    '''

    if images.ndim > 4:
        raise AssertionError('Augmentation will only work with 2D images')

    new_images = []
    new_labels = []
    num_images = images.shape[0]
    rows = images.shape[1]
    cols = images.shape[2]

    for ii in range(num_images):

        img = images[ii, ...]
        lbl = labels[ii, ...]

        # ROTATE
        angles = (-40, 40)
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
        #Float or [lower, upper].Range for random zoom.
        #If a float, `[lower, upper] = [1 - zoom_range, 1 + zoom_range]`
        zx, zy = np.random.uniform(1-0.05, 1+0.05, 2)

        # RANDOM HORIZONTAL FLIP
        flip_horizontal = (np.random.random() < 0.5)

        # RANDOM VERTICAL FLIP
        flip_vertical = (np.random.random() < 0.5)

        img, lbl = apply_affine_transform(img, lbl, rows=rows, cols=cols,
                                          theta=theta, tx=tx, ty=ty,
                                          zx=zx, zy=zy,
                                          fill_mode='nearest',
                                          order=1)

        if flip_horizontal:
            img = flip_axis(img, 1)
            lbl = flip_axis(lbl, 1)

        if flip_vertical:
            img = flip_axis(img, 0)
            lbl = flip_axis(lbl, 0)

        new_images.append(img)
        new_labels.append(lbl)

    sampled_image_batch = np.asarray(new_images)
    sampled_label_batch = np.asarray(new_labels)

    return sampled_image_batch, sampled_label_batch


def iterate_minibatches(images, labels, batch_size, augment_batch=False, expand_dims=True):
    '''
    Function to create mini batches from the dataset of a certain batch size
    :param images: input data shape (N, W, H)
    :param labels: label data
    :param batch_size: batch size (Int)
    :param augment_batch: should batch be augmented?, Boolean (default: False)
    :param expand_dims: adding a dimension, Boolean (default: True)
    :return: mini batches
    '''
    random_indices = np.arange(images.shape[0])
    np.random.shuffle(random_indices)
    n_images = images.shape[0]
    for b_i in range(0, n_images, batch_size):

        if b_i + batch_size > n_images:
            continue

        batch_indices = np.sort(random_indices[b_i:b_i + batch_size])
        X = images[batch_indices, ...]
        y = labels[batch_indices, ...]

        if expand_dims:
            X = X[..., np.newaxis]  # array of shape [minibatch, X, Y, nchannels]
            y = y[..., np.newaxis]

        if augment_batch:
            X, y = augmentation_function(X, y)

        yield X, y


def do_eval(images, labels, batch_size, augment_batch=False, expand_dims=True):
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
    for batch in iterate_minibatches(images,
                                     labels,
                                     batch_size,
                                     augment_batch,
                                     expand_dims):
        x, y = batch
        if y.shape[0] < batch_size:
            continue

        val_hist = model.test_on_batch(x, y)
        if history == []:
            history.append(val_hist)
        else:
            history[0] = [x + y for x, y in zip(history[0], val_hist)]
        num_batches += 1

    for i in range(len(history[0])):
        history[0][i] /= num_batches

    return history[0]


def print_txt(output_dir, stringa):
    out_file = os.path.join(output_dir, 'summary_report.txt')
    with open(out_file, "a") as text_file:
        text_file.writelines(stringa)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PATH
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
log_root = 'D:\GRASSO\logdir'
experiment_name = 'ConvMixUnet'
forceoverwrite = True

out_fold = os.path.join(log_root, experiment_name)

if not tf.io.gfile.exists(out_fold) or forceoverwrite:
    try:
        shutil.rmtree(out_fold)
    except:
        pass
    tf.io.gfile.makedirs(out_fold)
    out_file = os.path.join(out_fold, 'summary_report.txt')
    with open(out_file, "w") as text_file:
        text_file.write('\n--------------------------------------------------------------------------\n')
        text_file.write('Model summary')
        text_file.write('\n-----------------------------------------------------------------------------\n')
print_txt(out_fold, ['\nExperiment_name %s' % experiment_name])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
LOAD DATA
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
logging.info('\nLoading data...')
data = h5py.File(os.path.join('D:\GRASSO\data', 'train.hdf5'), 'r')
train_img = data['img_raw'][()]
train_label = data['mask'][()]
data.close()

data = h5py.File(os.path.join('D:\GRASSO\data', 'val.hdf5'), 'r')
val_img = data['img_raw'][()]
val_label = data['mask'][()]
data.close()

with open(out_file, "a") as text_file:
    text_file.write('\n----- Data summary -----')
print_txt(out_fold, ['\nTraining Images size: %s %s %s' % (train_img.shape[0], train_img.shape[1], train_img.shape[2])])
print_txt(out_fold, ['\nTraining Images shape: %s' % train_img.dtype])
print_txt(out_fold, ['\nValidation Images size: %s %s %s' % (val_img.shape[0], val_img.shape[1], val_img.shape[2])])
print_txt(out_fold, ['\nValidation Images shape: %s' % val_img.dtype])

if len(train_img) != len(train_label):
    raise Exception(
        'ERROR number of file is not equal in train (%s) and labels (%s)' % (len(train_img), len(train_label)))
if len(val_img) != len(val_label):
    raise Exception('ERROR number of file is not equal in train (%s) and labels (%s)' % (len(val_img), len(val_label)))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
NORMALIZATION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
for i in range(len(train_img)):
    train_img[i] = normalize_image(train_img[i])
for i in range(len(val_img)):
    val_img[i] = normalize_image(val_img[i])

train_img = train_img.astype('float32')
train_label = train_label.astype('float32')
val_img = val_img.astype('float32')
val_label = val_label.astype('float32')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
HYPERPARAMETERS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
batch_size = 4
epochs = 400
curr_lr = 0.001

with open(out_file, "a") as text_file:
    text_file.write('\n----- HYPERPARAMETERS -----')
print_txt(out_fold, ['\nbatch_size: %s' % batch_size])
print_txt(out_fold, ['\nepochs: %s' % epochs])
print_txt(out_fold, ['\ncurr_lr: %s\n\n' % curr_lr])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
LOADING MODEL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('\nCreating and compiling model...')
model = model_structure.ConvMixUnet(n_filt=64)

with open(out_file, 'a') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
# plot_model(model, to_file=os.path.join(log_dir,'model_plot.png'), show_shapes=True, show_layer_names=True)

opt = tf.keras.optimizers.Adam(learning_rate=curr_lr)
model.compile(optimizer=opt, loss=losses.focal_tversky_loss(), metrics=[losses.dice_coef])
print('Model prepared...')

if os.path.exists(os.path.join(out_fold, 'model_weights.h5')):
    print('\nLoading saved weights...')
    model.load_weights(os.path.join(out_fold, 'model_weights.h5'))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TRAINING MODEL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('Start training...')

best_val_loss = float('inf')
best_val_dice = float(0)
step = 0
no_improvement_counter = 0
name_metric = ['loss', 'dice_coef']
train_history = {}  # It records training metrics for each epoch
val_history = {}  # It records validation metrics for each epoch
lr_hist = []

for epoch in range(epochs):
    print('Epoch %d/%d' % (epoch + 1, epochs))
    temp_hist = {}
    for batch in iterate_minibatches(train_img,
                                     train_label,
                                     batch_size=batch_size,
                                     augment_batch=True,
                                     expand_dims=True):
        x, y = batch
        # TEMPORARY HACK (to avoid incomplete batches)
        if y.shape[0] < batch_size:
            step += 1
            continue

        hist = model.train_on_batch(x, y)
        if temp_hist == {}:
            for m_i in range(len(model.metrics_names)):
                temp_hist[model.metrics_names[m_i]] = []
        for key, i in zip(temp_hist, range(len(temp_hist))):
            temp_hist[key].append(hist[i])

        if (step + 1) % 40 == 0:
            print('step: %d, %s: %.3f, %s: %.3f' %
                  (step + 1, model.metrics_names[0], hist[0], model.metrics_names[1], hist[1]))

        step += 1  # fine batch

    # end epoch
    for key in temp_hist:
        temp_hist[key] = sum(temp_hist[key]) / len(temp_hist[key])

    logging.info('Training data Eval:')
    for m_k in range(len(model.metrics_names)):
        logging.info(str(model.metrics_names[m_k] + ': %.3f') % temp_hist[model.metrics_names[m_k]])

    if train_history == {}:
        for m_i in range(len(model.metrics_names)):
            train_history[model.metrics_names[m_i]] = []
    for key in train_history:
        train_history[key].append(temp_hist[key])

    # save learning rate history
    lr_hist.append(curr_lr)

    # evaluate the model against the validation set
    logging.info('Validation Data Eval:')
    val_hist = do_eval(val_img, val_label,
                       batch_size=batch_size,
                       augment_batch=False,
                       expand_dims=True)

    if val_history == {}:
        for m_i in range(len(model.metrics_names)):
            val_history[model.metrics_names[m_i]] = []
    for key, ii in zip(val_history, range(len(val_history))):
        val_history[key].append(val_hist[ii])

    # save best model
    if val_hist[1] > best_val_dice:
        no_improvement_counter = 0
        print('val_dice improved from %.3f to %.3f, saving model to weights-improvement' % (
            best_val_dice, val_hist[1]))
        best_val_dice = val_hist[1]
        model.save(os.path.join(out_fold, 'model_weights.h5'))
        # model.save_weights(os.path.join(log_dir, 'model_weights.h5'))
    else:
        no_improvement_counter += 1
        print('val_dice did not improve for %d epochs' % no_improvement_counter)

    # ReduceLROnPlateau
    if no_improvement_counter % 6 == 0 and no_improvement_counter != 0:
        curr_lr = curr_lr * 0.2
        if curr_lr < 1e-6:
            curr_lr = 1e-4
        K.set_value(model.optimizer.learning_rate, curr_lr)
        logging.info('Current learning rate: %.6f' % curr_lr)

    # EarlyStopping
    if no_improvement_counter > 48:  # Early stop if val loss does not improve after n epochs
        logging.info('Early stop at epoch {}.\n'.format(str(epoch + 1)))
        break

print('\nModel correctly trained and saved')

# plot history (loss and metrics)
plt.figure(figsize=(8, 8))
plt.grid(False)
plt.title("Learning curve LOSS", fontsize=20)
plt.plot(train_history["loss"], label="Train loss")
plt.plot(val_history["loss"], label="Val loss")
p = np.argmin(val_history["loss"])
plt.plot(p, val_history["loss"][p], marker="x", color="r", label="best model")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend();
plt.savefig(os.path.join(out_fold, 'Loss'), dpi=300)
plt.close()

plt.figure(figsize=(8, 8))
plt.grid(False)
plt.title("Dice Coefficient", fontsize=20)
plt.plot(train_history["dice_coef"], label="Train dice")
plt.plot(val_history["dice_coef"], label="Val dice")
p = np.argmax(val_history["dice_coef"])
plt.plot(p, val_history["dice_coef"][p], marker="x", color="r", label="best model")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Dice", fontsize=16)
plt.legend();
plt.savefig(os.path.join(out_fold, 'dice_coef'), dpi=300)
plt.close()

# plot learning rate
plt.figure(figsize=(8, 8))
plt.grid(False)
plt.title("Model learning rate", fontsize=20)
plt.plot(lr_hist)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("LR", fontsize=16)
plt.savefig(os.path.join(out_fold, 'LR'), dpi=300)
plt.close()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TESTING AND EVALUATING THE MODEL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('-' * 50)
print('Testing...')
print('-' * 50)
gt_exists = 'True'
test_path = os.path.join(out_fold, 'predictions')

if not tf.io.gfile.exists(test_path):
    tf.io.gfile.makedirs(test_path)
data_file_path = os.path.join(test_path, 'pred.hdf5')
out_pred_data = h5py.File(data_file_path, "w")

data = h5py.File(os.path.join('D:\GRASSO\data', 'test.hdf5'), 'r')
test_img = data['img_raw'][()]
test_label = data['mask'][()]
test_paz = data['paz'][()]
test_px = data['pixel_size'][()]
data.close()

with open(out_file, "a") as text_file:
    text_file.write('\n----- test Data summary -----')
print_txt(out_fold, ['\nTesting Images size: %s %s %s' % (test_img.shape[0], test_img.shape[1], test_img.shape[2])])
print_txt(out_fold, ['\nTesting Images shape: %s' % test_img.dtype])

print('Loading saved weights...')
model = tf.keras.models.load_model(os.path.join(out_fold, 'model_weights.h5'),
                                   custom_objects={'loss_function': losses.focal_tversky_loss(), 'dice_coef': losses.dice_coef})

RAW = []
PRED = []
PAZ = []
MASK = []
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
        RAW.append(img)
        PAZ.append(paz)
        PIXEL.append(test_px[ii])
        if gt_exists:
            MASK.append(test_label[ii])

        img = normalize_image(img)
        img = np.float32(img)
        x = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
        mask_out = model.predict(x)
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
if gt_exists:
    out_pred_data.create_dataset('mask', [n_file] + [160, 160], dtype=np.uint8)

for i in range(n_file):
    out_pred_data['img_raw'][i, ...] = RAW[i]
    out_pred_data['pred'][i, ...] = PRED[i]
    out_pred_data['paz'][i, ...] = PAZ[i]
    out_pred_data['pixel_size'][i, ...] = PIXEL[i]
    if gt_exists:
        out_pred_data['mask'][i, ...] = MASK[i]

out_pred_data.close()
logging.info('Average time per volume: %f' % (total_time / total_volumes))
print_txt(out_fold, ['\nAverage time per volume: %f' % (total_time / total_volumes)])

if gt_exists:
    metrics.main(test_path)
