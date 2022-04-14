from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf


# Helper function to enable loss function to be flexibly used for 
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5:
        return [1, 2, 3]
    # Two dimensional
    # (Batch, H, W, classes)
    elif len(shape) == 4:
        return [1, 2]
    # Exception - Unknown
    else:
        raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


################################
#           Dice loss          #
################################
def dice_loss(delta=0.5, smooth=0.000001):
    """Dice loss originates from Sørensen–Dice coefficient, which is a statistic developed in 1940s to gauge the similarity between two samples.
    
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """

    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1 - y_pred), axis=axis)
        fp = K.sum((1 - y_true) * y_pred, axis=axis)
        # Calculate Dice score
        dice_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
        # Average class scores
        dice_loss = K.mean(1 - dice_class)

        return dice_loss

    return loss_function


################################
#       Dice coefficient       #
################################
def dice_coef(y_true, y_pred):
    """The Dice similarity coefficient, also known as the Sørensen–Dice index or simply Dice coefficient, 
    is a statistical tool which measures the similarity between two sets of data.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
        0.5 dice_coeff otherwise tversky_index
        By assigning a greater weight to FN, recall is improved leading to a better balance of precision and recall.
        delta > 0.5 improve Recall
        delta < 0.5 improve precision
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """
    delta = 0.5
    smooth = 0.000001
    axis = identify_axis(y_true.get_shape())
    # Calculate true positives (tp), false negatives (fn) and false positives (fp)   
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1 - y_pred), axis=axis)
    fp = K.sum((1 - y_true) * y_pred, axis=axis)
    dice_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
    # Average class scores
    # with softmax dice_class shape [1,2] binary classification  --> return K.mean(dice_class)
    # with sigmoid dice_class shape [1]  --> return dice_class
    return K.mean(dice_class)


'''
def dice_coef(y_true, y_pred):
    smooth = 0.000001
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
'''


################################
#         Tversky loss         #
################################
def tversky_loss(delta=0.7, smooth=0.000001):
    """Tversky loss function for image segmentation using 3D fully convolutional deep networks
	Link: https://arxiv.org/abs/1706.05721
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
        > 0.5 greater weight to false negatives
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """

    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)   
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1 - y_pred), axis=axis)
        fp = K.sum((1 - y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
        # Average class scores
        tversky_loss = K.mean(1 - tversky_class)

        return tversky_loss

    return loss_function


################################
#          Combo loss          #
################################
def combo_loss(alpha=0.5):
    """Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation
    Link: https://arxiv.org/abs/1805.02798
    Parameters
    ----------
    alpha : float, optional
        controls weighting of dice and cross-entropy loss., by default 0.5
    """

    def loss_function(y_true, y_pred):
        dice = dice_loss()(y_true, y_pred)
        axis = identify_axis(y_true.get_shape())
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        # cross_entropy = K.binary_crossentropy(y_true, y_pred)
        # axis_to_reduce = range(1, K.ndim(cross_entropy))
        # cross_entropy = K.mean(x=cross_entropy, axis=axis_to_reduce)
        cross_entropy = K.mean(K.sum(cross_entropy, axis[-1]))

        # cross_entropy = K.mean(K.sum(cross_entropy, axis=[-1]))
        combo_loss = (alpha * cross_entropy) + ((1 - alpha) * dice)

        return combo_loss

    return loss_function


################################
#      Focal Tversky loss      #
################################
def focal_tversky_loss(delta=0.7, gamma=0.75, smooth=0.000001):
    """A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
    Link: https://arxiv.org/abs/1810.07842
    Parameters
    ----------
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """

    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1 - y_pred), axis=axis)
        fp = K.sum((1 - y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
        # Average class scores
        focal_tversky_loss = K.mean(K.pow((1 - tversky_class), gamma))

        return focal_tversky_loss

    return loss_function


################################
#          Focal loss          #
################################
def focal_loss(alpha=None, beta=None, gamma_f=2.):
    """Focal loss is used to address the issue of the class imbalance problem. A modulation term applied to the Cross-Entropy loss function.
    Parameters
    ----------
    alpha : float, optional
        controls relative weight of false positives and false negatives. Beta > 0.5 penalises false negatives more than false positives, by default None
    gamma_f : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 2.
    """

    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        if alpha is not None:
            alpha_weight = np.array(alpha, dtype=np.float32)
            focal_loss = alpha_weight * K.pow(1 - y_pred, gamma_f) * cross_entropy
        else:
            focal_loss = K.pow(1 - y_pred, gamma_f) * cross_entropy

        focal_loss = K.mean(K.sum(focal_loss, axis=[-1]))
        return focal_loss

    return loss_function


################################
#       Symmetric Focal loss      #
################################
def symmetric_focal_loss(delta=0.7, gamma=2.):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    """
    def loss_function(y_true, y_pred):

        axis = identify_axis(y_true.get_shape())

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        #calculate losses separately for each class
        back_ce = K.pow(1 - y_pred[:,:,:,0], gamma) * cross_entropy[:,:,:,0]
        back_ce =  (1 - delta) * back_ce

        fore_ce = K.pow(1 - y_pred[:,:,:,1], gamma) * cross_entropy[:,:,:,1]
        fore_ce = delta * fore_ce

        loss = K.mean(K.sum(tf.stack([back_ce, fore_ce],axis=-1),axis=-1))

        return loss

    return loss_function


#################################
# Symmetric Focal Tversky loss  #
#################################
def symmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

        #calculate losses separately for each class, enhancing both classes
        back_dice = (1-dice_class[:,0]) * K.pow(1-dice_class[:,0], -gamma)
        fore_dice = (1-dice_class[:,1]) * K.pow(1-dice_class[:,1], -gamma)

        # Average class scores
        loss = K.mean(tf.stack([back_dice,fore_dice],axis=-1))
        return loss

    return loss_function


################################
#     Asymmetric Focal loss    #
################################
def asymmetric_focal_loss(delta=0.7, gamma=2.):
    def loss_function(y_true, y_pred):
        """For Imbalanced datasets
        Parameters
        ----------
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.7
        gamma : float, optional
            Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
        """
        axis = identify_axis(y_true.get_shape())

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        #calculate losses separately for each class, only suppressing background class
        back_ce = K.pow(1 - y_pred[:,:,:,0], gamma) * cross_entropy[:,:,:,0]
        back_ce =  (1 - delta) * back_ce

        fore_ce = cross_entropy[:,:,:,1]
        fore_ce = delta * fore_ce

        loss = K.mean(K.sum(tf.stack([back_ce, fore_ce],axis=-1),axis=-1))

        return loss

    return loss_function

#################################
# Asymmetric Focal Tversky loss #
#################################
def asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

        #calculate losses separately for each class, only enhancing foreground class
        back_dice = (1-dice_class[:,0])
        fore_dice = (1-dice_class[:,1]) * K.pow(1-dice_class[:,1], -gamma)

        # Average class scores
        loss = K.mean(tf.stack([back_dice,fore_dice],axis=-1))
        return loss

    return loss_function


###########################################
#      Symmetric Unified Focal loss       #
###########################################
def sym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def loss_function(y_true,y_pred):
      symmetric_ftl = symmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true,y_pred)
      symmetric_fl = symmetric_focal_loss(delta=delta, gamma=gamma)(y_true,y_pred)
      if weight is not None:
        return (weight * symmetric_ftl) + ((1-weight) * symmetric_fl)
      else:
        return symmetric_ftl + symmetric_fl

    return loss_function

###########################################
#      Asymmetric Unified Focal loss      #
###########################################
def asym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def loss_function(y_true,y_pred):

        asymmetric_ftl = asymmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true,y_pred)
        asymmetric_fl = asymmetric_focal_loss(delta=delta, gamma=gamma)(y_true,y_pred)
        if weight is not None:
            return (weight * asymmetric_ftl) + ((1-weight) * asymmetric_fl)
        else:
            return asymmetric_ftl + asymmetric_fl

    return loss_function


###########################################
#      hybrid loss      #
###########################################
def hybrid_loss(weight=0.5, delta=0.7, gamma=0.75, smooth=0.000001):
    def loss_function(y_true, y_pred):
        focal_tversky = focal_tversky_loss(delta=delta, gamma=gamma, smooth=smooth)(y_true,y_pred)
        focal = focal_loss()(y_true,y_pred)
        if weight is not None:
            return (weight * focal_tversky) + ((1 - weight) * focal)
        else:
            return focal_tversky + focal
    return loss_function


################################
#      Focal Tversky loss      #
################################
def focal_tversky_loss(delta=0.7, gamma=0.75, smooth=0.000001):
    """A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
    Link: https://arxiv.org/abs/1810.07842
    Parameters
    ----------
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """

    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1 - y_pred), axis=axis)
        fp = K.sum((1 - y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
        # Average class scores
        focal_tversky_loss = K.mean(K.pow((1 - tversky_class), gamma))

        return focal_tversky_loss

    return loss_function

def convert_to_logits(y_pred):
    """
    Converting output of sigmoid to logits.
    :param y_pred: Predictions after sigmoid (<BATCH_SIZE>, shape=(None, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1)).
    :return: Logits (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
    """
    # To avoid unwanted behaviour of log operation
    y_pred = K.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    return K.log(y_pred / (1 - y_pred))


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))
