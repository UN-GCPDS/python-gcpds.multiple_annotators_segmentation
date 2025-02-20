import tensorflow as tf 

def specificity_metric(y_true, y_pred, axis=(1, 2), smooth=1e-7, return_mean=True):
    """
    Computes the specificity (true negative rate) between predicted and ground-truth segmentation masks.

    Specificity measures the proportion of actual negatives correctly identified by the model:
    TN / (TN + FP), where TN = true negatives and FP = false positives.

    This implementation does not binarize the input tensors; the user should ensure that 
    `y_true` and `y_pred` are binary (0 or 1) tensors with the same shape.

    Args:
        y_true (tf.Tensor): Ground truth segmentation masks 
            (shape: [batch_size, height, width, channels]).
        y_pred (tf.Tensor): Predicted segmentation masks 
            (shape: [batch_size, height, width, channels]).
        axis (tuple, optional): Spatial axes to reduce (typically height and width). 
            Defaults to (1, 2).
        smooth (float, optional): This parameter is currently unused. The denominator uses 
            `tf.keras.backend.epsilon()` for numerical stability. Defaults to 1e-7.
        return_mean (bool, optional): If True, returns the mean specificity across all samples and classes. 
            If False, returns per-sample or per-class scores. Defaults to True.

    Returns:
        tf.Tensor:
            - If `return_mean=True`: A scalar tensor representing the overall mean specificity.
            - If `return_mean=False`: A tensor containing specificity scores. Shape depends on input:
                - For single-channel inputs: [batch_size]
                - For multi-channel inputs: [channels, batch_size] (requires post-processing for per-sample metrics)

    Example:
        >>> y_true = tf.constant([[[[1], [0]], [[0], [1]]]], dtype=tf.float32)  # Shape: (1, 2, 2, 1)
        >>> y_pred = tf.constant([[[[0.9], [0.2]], [[0.1], [0.8]]]], dtype=tf.float32)  # Shape: (1, 2, 2, 1)
        >>> # Convert y_pred to binary (e.g., using a threshold)
        >>> y_pred_binary = tf.where(y_pred > 0.5, 1.0, 0.0)
        >>> specificity_score = specificity_metric(y_true, y_pred_binary, return_mean=True)
        >>> print(specificity_score)  # Output: ~0.5 (example value, depends on data)

        >>> per_sample_scores = specificity_metric(y_true, y_pred_binary, return_mean=False)
        >>> print(per_sample_scores)  # Output: [0.5] (per-sample score)

    Note:
        - This implementation assumes binary segmentation masks (single-channel or multi-channel with independent classes).
        - The function does not binarize input tensors; users must preprocess `y_true` and `y_pred` to binary values (0/1).
        - When `return_mean=False`, the output shape depends on the number of input channels:
            - Single-channel inputs produce [batch_size] shape
            - Multi-channel inputs produce [channels, batch_size] shape (requires transposing or reshaping for per-sample metrics)
        - The `smooth` parameter is unused; the denominator uses `tf.keras.backend.epsilon()` to avoid division by zero.
    """
    true_negatives = tf.reduce_sum((1 - y_true) * (1 - y_pred), axis=axis)
    actual_negatives = tf.reduce_sum(1 - y_true, axis=axis)
    specificity = true_negatives / (actual_negatives + tf.keras.backend.epsilon())
    if return_mean:
        return tf.reduce_mean(specificity)
    else:
        return tf.squeeze(tf.transpose(specificity, perm=[1, 0]), axis=0)
