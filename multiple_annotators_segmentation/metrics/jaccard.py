import tensorflow as tf 
from tensorflow.keras.metrics import Metric

def jaccard_metric(y_true, y_pred, axis=(1, 2), smooth=1e-5):
    """Computes the Jaccard similarity coefficient as a metric for semantic segmentation.

    The Jaccard similarity coefficient, also known as the Intersection over Union (IoU),
    measures the similarity between two sets by comparing their intersection to their union.
    In the context of semantic segmentation, it quantifies the overlap between the ground
    truth segmentation masks and the predicted segmentation masks.

    Parameters:
        y_true (tensor): Ground truth segmentation masks.
        y_pred (tensor): Predicted segmentation masks.
        axis (tuple of int): Axes along which to compute sums. Defaults to (1, 2).
        smooth (float): A small smoothing parameter to avoid division by zero. Defaults to 1e-5.

    Returns:
        A tensor representing the mean Jaccard similarity coefficient.

    """
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_pred, axis=axis) - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(jaccard)
