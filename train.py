import numpy as np 
import os
import h5py
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
from scipy import ndimage
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import logging
import model_structure
import losses
logging.basicConfig(
    level=logging.INFO # allow DEBUG level messages to pass through the logger
    )

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
    img_o = (img_o-img_o.min())/(img_o.max()-img_o.min())
    return img_o
def myprint(s):
    with open('modelsummary.txt','w+') as f:
        print(s, file=f)
  

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x
    

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 - 0.5
    o_y = float(y) / 2 - 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix
    
    
def apply_affine_transform(img, lbl, rows, cols, theta=0, tx=0, ty=0,
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
            img[:,:,channel],
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=0.0) for channel in range(img.shape[-1])]
        img = np.stack(channel_images, axis=2)

        channel_images = [ndimage.interpolation.affine_transform(
            lbl[:,:,channel],
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

        img = images[ii,...]
        lbl = labels[ii,...]

        # ROTATE
        angles = (-30,30)
        theta = np.random.uniform(angles[0], angles[1])

        #RANDOM WIDTH SHIFT
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
        
        #RANDOM HEIGHT SHIFT
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
        
        #RANDOM HORIZONTAL FLIP
        flip_horizontal = (np.random.random() < 0.5)
              
        #RANDOM VERTICAL FLIP
        flip_vertical = (np.random.random() < 0.5)
        
        img, lbl = apply_affine_transform(img, lbl, rows=rows, cols=cols,
                                          theta=theta, tx=tx, ty=ty,
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
    for b_i in range(0,n_images,batch_size):

        if b_i + batch_size > n_images:
            continue
        
        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])
        X = images[batch_indices, ...]
        y = labels[batch_indices, ...]

        if expand_dims:        
            X = X[...,np.newaxis]   #array of shape [minibatch, X, Y, nchannels]
            y = y[...,np.newaxis]

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
        
        val_hist = model.test_on_batch(x,y)
        if history == []:
            history.append(val_hist)
        else:
            history[0] = [x + y for x, y in zip(history[0], val_hist)]
        num_batches += 1

    for i in range(len(history[0])):
        history[0][i] /= num_batches
    
    return history[0]

"""
Load data train
"""
logging.info('\nLoading data...')
data = h5py.File('train.hdf5', 'r')

train_img = data['img_raw'][()]
train_label = data['mask'][()]
train_up = data['img_up'][()]
train_down = data['img_down'][()]
train_left = data['img_left'][()]
train_right = data['img_right'][()]
data.close()

data = h5py.File('val.hdf5', 'r')

val_img = data['img_raw'][()]
val_label = data['mask'][()]
val_up = data['img_up'][()]
val_down = data['img_down'][()]
val_left = data['img_left'][()]
val_right = data['img_right'][()]
data.close()

logging.info('\nData summary:')
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

"""
Pre process
"""
for i in range(train_img.shape[0]):
  train_img[i] = standardize_image(train_img[i])
for i in range(val_img.shape[0]):
  val_img[i] = standardize_image(val_img[i])

img_train = train_img.astype('float32')
mask_train = train_label.astype('float32')
img_val = val_img.astype('float32')
mask_val = val_label.astype('float32')

# only for softmax
#mask_train = tf.keras.utils.to_categorical(mask_train, num_classes=2)
#mask_val = tf.keras.utils.to_categorical(mask_val, num_classes=2)

"""
Hyperparameters
"""
batch_size = 2
epochs = 10
#decay = len(train_img)//batch_size
curr_lr = 1e-3
pretrained_weights = None # 'C://..../model_weight.h5'

"""
Model
"""
print('\nCreating and compiling model...')
model = model_structure.Unet()
with open('modelsummary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.summary()

class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, curr_lr):
    self.curr_lr = curr_lr
  @tf.function
  def __call__(self, step):
    if (step+1) % decay == 0:
        self.curr_lr = self.curr_lr * 0.97
    return self.curr_lr
  #def get_config(self):
  #  config = {
  #      'curr_lr': self.curr_lr
  #  }
  #  return config
    
#opt = tf.keras.optimizers.Adam(learning_rate=MyLRSchedule(curr_lr))
opt = tf.keras.optimizers.Adam(learning_rate=curr_lr)

model.compile(optimizer=opt, loss=losses.combo_loss(), 
              metrics = [dice_coef, 'accuracy', precision, recall, f1score])

if (pretrained_weights):
    print('\nLoading saved weights...')
    print('Using {0} pretrained weights'.format(pretrained_weights))
    model.load_weights(pretrained_weights)

"""
Define callbacks

stopping = EarlyStopping(patience=16,verbose=1, monitor='val_dice_coef', mode='max')
checkpoint = ModelCheckpoint(filepath='model_weight.h5',
                             monitor='val_dice_coef',
                             mode='max',
                             save_best_only=True,
                             verbose=1)
def scheduler(epoch, lr):
  if epoch == 0:
    return lr
  else:
    return lr * 0.97
callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
"""

logging.info('\nFitting model...')

'''
results =model.fit(img_train, mask_train, batch_size=batch_size, epochs=epochs,
                   validation_data=(img_val, mask_val), 
                   callbacks=[stopping, checkpoint, callback],
                   shuffle=True)
'''

log_dir = ''
best_val_loss = float('inf')
best_val_dice = float(0)
step = 0
no_improvement_counter = 0
name_metric = ['loss', 'dice_coef', 'accuracy', 'precision', 'recall', 'f1score']
train_history  = {}   #It records training metrics for each epoch
val_history = {}    #It records validation metrics for each epoch
lr_hist = []

logging.info('Start training...')
for epoch in range(epochs):
    logging.info('Epoch {}/{}:'.format(str(epoch+1), str(epochs)))
    temp_hist = {}
    for batch in iterate_minibatches(img_train,
                                     mask_train,
                                     batch_size=batch_size,
                                     augment_batch=True,
                                     expand_dims=True):
        x, y = batch

        #TEMPORARY HACK (to avoid incomplete batches)
        if y.shape[0] < batch_size:
            step += 1
            continue

        hist = model.train_on_batch(x,y)
        if temp_hist == {}:
            for m_i in range(len(model.metrics_names)):
                temp_hist[model.metrics_names[m_i]] = []
        for key, i in zip(temp_hist, range(len(temp_hist))):
                    temp_hist[key].append(hist[i])
        
        if (step + 1) % 20 == 0:
            logging.info(str('step: %d '+name_metric[0]+': %.3f '+name_metric[1]+': %.3f '+name_metric[2]+': %.3f '
            +name_metric[3]+': %.3f '+name_metric[4]+': %.3f '+name_metric[5]+': %.3f') % 
                         (step+1, hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]))
        
        step += 1  #fine batch
    
    for key in temp_hist:
        temp_hist[key] = sum(temp_hist[key])/len(temp_hist[key])

    for m_k in range(len(model.metrics_names)):
        logging.info(str(model.metrics_names[m_k]+': %.3f') % temp_hist[model.metrics_names[m_k]])

    if train_history == {}:
        for m_i in range(len(model.metrics_names)):
            train_history[model.metrics_names[m_i]] = []
    for key in history:
        train_history[key].append(temp_hist[key])

    #save learning rate history
    lr_hist.append(curr_lr)

    #decay learning rate
    curr_lr = curr_lr * 0.97
    K.set_value(model.optimizer.learning_rate, curr_lr)
    logging.info('Current learning rate: %f' % curr_lr)

    #evaluate the model against the validation set
    logging.info('Validation Data Eval:')
    val_hist = do_eval(img_val, mask_val,
                       batch_size=batch_size,
                       augment_batch=False,
                       expand_dims=True)

    if val_history == {}:
        for m_i in range(len(model.metrics_names)):
            val_history[model.metrics_names[m_i]] = []
    for key, ii in zip(val_history, range(len(val_history))):
        val_history[key].append(val_hist[ii])
    
    #save best model
    if val_hist[1] > best_val_dice:
        no_improvement_counter = 0
        logging.info(str('val_'+model.metrics_names[1]+' improved from %.3f to %.3f, saving model to weights-improvement') % (best_val_dice, val_hist[1]))
        best_val = val_hist[1]
        model.save(os.path.join(log_dir, 'model_weight.h5'))
    else:
        no_improvement_counter += 1
        logging.info('val_dice_coef did not improve for %d epochs' % no_improvement_counter)

    #EarlyStopping
    if no_improvement_counter > 30:  # Early stop if val loss does not improve after 30 epochs
        logging.info('Early stop at epoch {}.\n'.format(str(epoch+1)))
        break

print('\nModel correctly trained and saved')

#plot history (loss and metrics)
for key in train_history:
    plt.figure()
    plt.plot(train_history[key], label=key)
    plt.plot(val_history[key], label=str('val_'+key))
    plt.title(str('model '+key))
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(key)
    plt.show()
    plt.savefig(os.path.join(log_dir, str(key)+'.png'))
#plot learning rate
plt.figure()
plt.plot(lr_hist)
plt.title('model learning rate')
plt.xlabel('epoch')
plt.ylabel('learning rate')
plt.show()
plt.savefig(os.path.join(log_dir,'learning_rate.png'))
