import tensorflow as tf 
from tensorflow.keras.metrics import Metric

def specificity_metric(y_true, y_pred, axis=(1, 2), smooth=1e-5):
    """Computes the specificity as a metric for semantic segmentation.

    Specificity measures the proportion of actual negative cases that were correctly 
    identified as such. It is complementary to sensitivity (recall).

    Parameters:
        y_true (tensor): Ground truth binary labels.
        y_pred (tensor): Predicted probabilities or binary predictions.
        axis (tuple): Axes over which to perform reduction. Defaults to (1, 2).
        smooth (float): Smoothing term to avoid division by zero. Defaults to 1e-5.

    Returns:
        A tensor representing the specificity metric.

    """
    true_negatives = tf.reduce_sum((1 - y_true) * (1 - y_pred), axis=axis)
    actual_negatives = tf.reduce_sum(1 - y_true, axis=axis)
    specificity = true_negatives / (actual_negatives + tf.keras.backend.epsilon())
    return tf.reduce_mean(specificity)
