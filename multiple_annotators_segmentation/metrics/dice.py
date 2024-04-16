import tensorflow as tf 
from tensorflow.keras.metrics import Metric

def dice_metric(y_true, y_pred, axis=(1, 2), smooth=1e-5):
    """Computes the Dice coefficient metric for evaluating semantic segmentation.

    This function calculates the Dice coefficient metric, which measures the similarity 
    between ground truth and predicted segmentation masks.
    Parameters:
        y_true (tensor): Ground truth segmentation masks.
        y_pred (tensor): Predicted segmentation masks.
        axis (tuple of int): Axis along which to compute sums. Defaults to (1, 2).
        smooth (float): A smoothing parameter to avoid division by zero. Defaults to 1e-5.

    Returns:
        A scalar value representing the average Dice coefficient metric.

    """
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_pred, axis=axis)
    dice = (2. * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)
