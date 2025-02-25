import os
import numpy as np
import tensorflow as tf
from ..metrics import dice_metric, jaccard_metric, sensitivity_metric, specificity_metric

def performance_measures_crowd_seg_model(
    model,
    test_partition_dataset,
    config_model,
    target_class=None,
    save_results=True
):
    """
    Compute performance metrics for a crowd segmentation model.

    This function evaluates a model's performance by calculating the Dice coefficient,
    Jaccard index, sensitivity, and specificity across a test dataset. It handles
    both single-class and multi-class segmentation tasks efficiently by processing
    data in batches and optionally saving results to disk to conserve memory.

    Parameters:
    -----------
    model : tensorflow.keras.Model
        A trained TensorFlow/Keras model capable of generating segmentation masks.
        The model should accept input batches and return predictions in a format
        compatible with the segmentation task (e.g., logits or probabilities for
        each class).
    test_partition_dataset : tf.data.Dataset
        The dataset used for performance evaluation. It should yield batches of
        image data, ground-truth masks, annotator IDs, and original masks in the
        format (images, masks, id_annotator, orig_masks). The original masks are
        used to compute metrics against the model's predictions.
    target_class : int or None, optional (default=None)
        Specifies which class to evaluate:
        - If `int`, metrics are computed only for the class at the given index.
        - If `None`, metrics are computed for all classes (requires the model to
          return per-class predictions).
    save_results : bool, optional (default=True)
        If True, the computed metrics are saved as NumPy files (`.npy`), using the
        configuration parameters from `config_model` to generate unique filenames.

    Returns:
    --------
    None
        This function prints the average and standard deviation for each metric and
        optionally saves the results to disk. To capture these metrics programmatically,
        consider modifying the function to return the results or redirect the print
        statements.

    Raises:
    -------
    ValueError
        Raised if `target_class` is neither `None` nor an integer.

    Notes:
    ------
    - The `config_model` dictionary is expected to have keys like "Main_model",
      "Activation", and "Dataset" for constructing the filenames when saving results.
    - For large datasets, setting `save_results=True` and using small batch sizes
      is recommended to prevent memory overflow.
    - The performance metrics (Dice, Jaccard, sensitivity, specificity) are computed
      using functions like `dice_metric`, `jaccard_metric`, `sensitivity_metric`, and
      `specificity_metric`, which must be defined elsewhere in your code.
    - The `model.predict` method is called with the inputs `(images, id_annotator)`.

    """
    # Initialize empty lists to store metric results
    dice_results = []
    jaccard_results = []
    sensitivity_results = []
    specificity_results = []

    # Process each batch in the test dataset
    for images, masks, id_annotator, orig_masks in test_partition_dataset.take(len(test_partition_dataset)):
        # Generate model predictions
        y_pred, _ = model.predict((images, id_annotator))
        
        # Determine the class(es) to evaluate and compute metrics
        if target_class is not None:
            if isinstance(target_class, int):
                # Evaluate a specific class by slicing the predictions
                y_pred_class = y_pred[..., target_class:target_class+1]
                dice_batch = dice_metric(orig_masks, y_pred_class, return_mean=False)
                jaccard_batch = jaccard_metric(orig_masks, y_pred_class, return_mean=False)
                sensitivity_batch = sensitivity_metric(orig_masks, y_pred_class, return_mean=False)
                specificity_batch = specificity_metric(orig_masks, y_pred_class, return_mean=False)
            else:
                # Assume the model's prediction covers all classes
                y_pred_class = y_pred
                dice_batch = dice_metric(orig_masks, y_pred_class, return_mean=False)
                jaccard_batch = jaccard_metric(orig_masks, y_pred_class, return_mean=False)
                sensitivity_batch = sensitivity_metric(orig_masks, y_pred_class, return_mean=False)
                specificity_batch = specificity_metric(orig_masks, y_pred_class, return_mean=False)

            # Store results from the current batch
            dice_results.append(dice_batch.numpy().flatten())
            jaccard_results.append(jaccard_batch.numpy().flatten())
            sensitivity_results.append(sensitivity_batch.numpy().flatten())
            specificity_results.append(specificity_batch.numpy().flatten())
        else:
            # Raise an error if target_class is not specified correctly
            raise ValueError(f"'target_class' must be either an integer or None. Received: {target_class}")

        # Free memory by deleting unnecessary tensors immediately
        del y_pred

    # Convert lists of batches to flat NumPy arrays for analysis
    dice_results = np.concatenate(dice_results)
    jaccard_results = np.concatenate(jaccard_results)
    sensitivity_results = np.concatenate(sensitivity_results)
    specificity_results = np.concatenate(specificity_results)

    # Print performance summaries with 5 decimal places precision
    print("Model's performance metrics:")
    print(f"DICE Coefficient mean: {np.mean(dice_results):.5f}, standard deviation: {np.std(dice_results):.5f}")
    print(f"Jaccard Index mean: {np.mean(jaccard_results):.5f}, standard deviation: {np.std(jaccard_results):.5f}")
    print(f"Sensitivity mean: {np.mean(sensitivity_results):.5f}, standard deviation: {np.std(sensitivity_results):.5f}")
    print(f"Specificity mean: {np.mean(specificity_results):.5f}, standard deviation: {np.std(specificity_results):.5f}")

    # Save results to disk if requested
    if save_results:
        # Check if the directory exists, and if not, create it
        if not os.path.exists('./results'):
            os.makedirs('./results')
        # Generate filenames using the configuration parameters
        filename_base = f"./results/{config_model['Main_model']}_{config_model['Activation']}_{config_model['Dataset']}"
        np.save(f"{filename_base}_DICE.npy", dice_results)
        np.save(f"{filename_base}_Jaccard.npy", jaccard_results)
        np.save(f"{filename_base}_Sensitivity.npy", sensitivity_results)
        np.save(f"{filename_base}_Specificity.npy", specificity_results)

    # Cleanup intermediate results to free memory
    del dice_results, jaccard_results, sensitivity_results, specificity_results