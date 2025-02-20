import tensorflow as tf 

def jaccard_metric(y_true, y_pred, axis=(1, 2), smooth=1e-7, return_mean=True):
    """
    Computes the Jaccard similarity coefficient (Intersection over Union) between 
    predicted and ground-truth segmentation masks.

    The Jaccard index is a measure of overlap between two binary sets and is defined as:
    |A ∩ B| / |A ∪ B|, where A and B are the sets of pixels in the predicted and ground-truth masks.

    This implementation binarizes the input tensors using a threshold of 0.5 before calculation.

    Args:
        y_true (tf.Tensor): Ground truth segmentation masks 
            (shape: [batch_size, height, width, channels]).
        y_pred (tf.Tensor): Predicted segmentation masks 
            (shape: [batch_size, height, width, channels]).
        axis (tuple, optional): Spatial axes to reduce (typically height and width). 
            Defaults to (1, 2).
        smooth (float, optional): Small value added to numerator and denominator 
            to avoid division by zero. Defaults to 1e-7.
        return_mean (bool, optional): If True, returns the mean Jaccard score across all samples. 
            If False, returns per-sample scores. Defaults to True.

    Returns:
        tf.Tensor:
            - If `return_mean=True`: A scalar tensor representing the mean Jaccard score.
            - If `return_mean=False`: A tensor of shape [batch_size] containing 
              Jaccard scores for each sample.

    Example:
        >>> y_true = tf.constant([[[[1], [0]], [[0], [1]]]], dtype=tf.float32)
        >>> y_pred = tf.constant([[[[0.9], [0.2]], [[0.1], [0.8]]]], dtype=tf.float32)
        >>> jaccard_score = jaccard_metric(y_true, y_pred)
        >>> print(jaccard_score)  # Output: ~0.76  (scalar mean score)

        >>> per_sample_scores = jaccard_metric(y_true, y_pred, return_mean=False)
        >>> print(per_sample_scores)  # Output: [0.76] (per-sample score)

    Note:
        - This implementation assumes binary segmentation masks (single-channel or 
          multi-channel with independent classes).
        - The thresholding at 0.5 is hardcoded; adjust if your application uses different 
          binarization criteria.
        - The `axis` parameter should match the spatial dimensions of your input data 
          (e.g., (1, 2) for 2D images).

    """
    y_true = tf.cast(y_true, tf.float32)
    y_true = tf.where(y_true > 0.5, 1.0, 0.0)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.where(y_pred > 0.5, 1.0, 0.0)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_pred, axis=axis) - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    if return_mean:
        return tf.reduce_mean(jaccard)
    else:
        return tf.squeeze(tf.transpose(jaccard, perm=[1, 0]), axis=0)