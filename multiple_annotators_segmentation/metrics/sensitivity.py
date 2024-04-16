import tensorflow as tf 
from tensorflow.keras.metrics import Metric

def sensitivity_metric(y_true, y_pred, axis=(1, 2), smooth=1e-5):
    """Computes the sensitivity as a metric for semantic segmentation.

    Sensitivity, also known as true positive rate or recall, measures the proportion 
    of actual positives that are correctly identified by the model. It is computed 
    as the ratio of true positives to the sum of true positives and false negatives.

    Parameters:
        y_true (tensor): Ground truth labels.
        y_pred (tensor): Predicted probabilities or labels.
        axis (tuple): Axes over which to perform the reduction. Defaults to (1, 2).
        smooth (float): A small value added to the denominator to avoid division by zero. 
            Defaults to 1e-5.

    Returns:
        The sensitivity metric averaged over the specified axes.

    """
    true_positives = tf.reduce_sum(y_true * y_pred, axis=axis)
    actual_positives = tf.reduce_sum(y_true, axis=axis)
    sensitivity = true_positives / (actual_positives + tf.keras.backend.epsilon())
    return tf.reduce_mean(sensitivity)
