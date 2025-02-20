import tensorflow as tf

def noisy_label_loss_tf(pred, cms, labels, min_trace=False, alpha=0.1):
    """
    Computes the noisy label loss for segmentation tasks.

    This function calculates the loss considering the predicted segmentation maps,
    the crowd matrices (cms), and the ground truth labels. It supports two modes:
    'ce' for cross-entropy loss and 'mse' for mean squared error loss.
    Additionally, it includes a regularization term based on the trace of the
    crowd matrices, which can be minimized or maximized based on the `min_trace` flag.

    Args:
        pred (tf.Tensor): Predicted segmentation maps with shape (batch_size, num_classes, height, width).
        cms (tf.Tensor): Crowd matrices with shape (batch_size, num_classes^2, height, width).
        labels (tf.Tensor): Ground truth labels with shape (batch_size, height, width).
        min_trace (bool, optional): Whether to minimize or maximize the trace regularization. Defaults to False.
        alpha (float, optional): Weight for the trace regularization term. Defaults to 0.1.

    Returns:
        tf.Tensor: The computed loss tensor.
    """
    shape_pred = tf.shape(pred)
    pred = tf.reshape(pred, (shape_pred[0], shape_pred[1], shape_pred[2], shape_pred[3]))

    shape_cms = tf.shape(cms)
    cms = tf.reshape(cms, (shape_cms[0], shape_cms[1], shape_cms[2], shape_cms[3]))

    shape_labels = tf.shape(labels)
    labels = tf.reshape(labels, (shape_labels[0], shape_labels[1], shape_labels[2]))

    # Get dimensions
    b = tf.shape(pred)[0]
    c = tf.shape(pred)[1]
    h = tf.shape(pred)[2]
    w = tf.shape(pred)[3]

    # Reshape pred for matrix multiplication
    pred_norm = tf.reshape(tf.transpose(tf.reshape(pred, [b, c, h*w]), perm=[0, 2, 1]), [b*h*w, c, 1])

    # Reshape and normalize cm
    cm = tf.reshape(tf.reshape(tf.transpose(tf.reshape(cms, [b, c**2, h*w]), perm=[0, 2, 1]), [b * h * w, c * c]), [b * h * w, c, c])
    cm = cm / tf.reduce_sum(cm, axis=1, keepdims=True)

    # Compute noisy predictions
    pred_noisy = tf.matmul(cm, pred_norm)

    pred_noisy = tf.transpose(tf.reshape(tf.transpose(tf.reshape(tf.reshape(pred_noisy, [b*h*w, c]), [b, h*w, c]), perm=[0, 2, 1]), [b, c, h, w]), perm=[0, 2, 3, 1])

    # Create mask for ignore_index
    loss_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(tf.reshape(labels, [b, h, w]), pred_noisy)

    # Compute regularization (trace)
    regularisation = tf.reduce_sum(tf.linalg.trace(tf.transpose(tf.reduce_sum(cm, axis=0), perm=[1, 0]))) / tf.cast(b * h * w, tf.float32)
    regularisation = alpha * regularisation

    # Final loss
    if min_trace:
        loss = loss_ce + regularisation
    else:
        loss = loss_ce - regularisation

    return loss