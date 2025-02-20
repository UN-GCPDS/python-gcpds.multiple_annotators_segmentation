import tensorflow as tf 

def sensitivity_metric(y_true, y_pred, axis=(1, 2), smooth=1e-7, return_mean=True):
    """
    Computes the sensitivity (recall/true positive rate) between predicted and ground-truth segmentation masks.

    Sensitivity measures the proportion of actual positives correctly identified by the model:
    TP / (TP + FN), where TP = true positives and FN = false negatives.

    This implementation binarizes input tensors using a threshold of 0.5 before calculation.

    Args:
        y_true (tf.Tensor): Ground truth segmentation masks 
            (shape: [batch_size, height, width, channels]).
        y_pred (tf.Tensor): Predicted segmentation masks 
            (shape: [batch_size, height, width, channels]).
        axis (tuple, optional): Spatial axes to reduce (typically height and width). 
            Defaults to (1, 2).
        smooth (float, optional): Small value added to denominator to avoid division by zero. 
            Defaults to 1e-7.
        return_mean (bool, optional): If True, returns the mean sensitivity across all samples and classes. 
            If False, returns per-sample or per-class scores. Defaults to True.

    Returns:
        tf.Tensor:
            - If `return_mean=True`: A scalar tensor representing the overall mean sensitivity.
            - If `return_mean=False`: A tensor containing sensitivity scores. Shape depends on input:
                - For single-channel inputs: [batch_size]
                - For multi-channel inputs: [channels, batch_size] (requires post-processing for per-sample metrics)

    Example:
        >>> y_true = tf.constant([[[[1], [0]], [[0], [1]]]], dtype=tf.float32)  # Shape: (1, 2, 2, 1)
        >>> y_pred = tf.constant([[[[0.9], [0.2]], [[0.1], [0.8]]]], dtype=tf.float32)  # Shape: (1, 2, 2, 1)
        >>> sensitivity_score = sensitivity_metric(y_true, y_pred)
        >>> print(sensitivity_score)  # Output: ~0.83 (scalar mean score)

        >>> per_sample_scores = sensitivity_metric(y_true, y_pred, return_mean=False)
        >>> print(per_sample_scores)  # Output: [0.83] (per-sample score for single-channel input)

    Note:
        - This implementation assumes binary segmentation masks (single-channel or multi-channel with independent classes).
        - The thresholding at 0.5 is hardcoded; adjust if your application uses different binarization criteria.
        - When `return_mean=False`, the output shape depends on the number of input channels:
            - Single-channel inputs produce [batch_size] shape
            - Multi-channel inputs produce [channels, batch_size] shape (requires transposition/squeezing for per-sample metrics)
        - The `smooth` parameter helps handle cases with no actual positives (denominator becomes `smooth`).

    """
    y_true = tf.cast(y_true, tf.float32)
    y_true = tf.where(y_true > 0.5, 1.0, 0.0)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.where(y_pred > 0.5, 1.0, 0.0)
    true_positives = tf.reduce_sum(y_true * y_pred, axis=axis)
    actual_positives = tf.reduce_sum(y_true, axis=axis)
    sensitivity = true_positives / (actual_positives + smooth)
    if return_mean:
        return tf.reduce_mean(sensitivity)
    else:
        return tf.squeeze(tf.transpose(sensitivity, perm=[1, 0]), axis=0)