import os
import numpy as np
import tensorflow as tf
from ..metrics import dice_metric, jaccard_metric, sensitivity_metric, specificity_metric

def performance_measures_annot_harmony_model(
    model,
    test_partition_dataset,
    config_model,
    target_class=None,
    save_results=True
):
    """
    Evaluate the performance of an annotation harmony model on a test dataset.
    
    This function calculates key segmentation metrics (Dice coefficient, Jaccard index,
    sensitivity, and specificity) for a trained model over the entire test dataset.
    Results can be saved to disk for further analysis.
    
    Args:
        model: Trained segmentation model to evaluate
        test_partition_dataset: TensorFlow dataset containing test data with format
                               (images, masks, original_masks)
        config_model: Dictionary containing model configuration parameters used for
                     naming saved results
        target_class: Integer specifying which class to evaluate, or None to evaluate
                     all classes (default: None)
        save_results: Boolean flag indicating whether to save results to disk (default: True)
        
    Returns:
        None: Prints performance metrics and optionally saves results to disk
        
    Raises:
        ValueError: If target_class is not correctly specified
    """
    # Initialize empty lists to store metric results
    dice_results = []
    jaccard_results = []
    sensitivity_results = []
    specificity_results = []
    
    # Process each batch in the test dataset
    for images, masks, orig_masks in test_partition_dataset.take(len(test_partition_dataset)):
        # Generate model predictions
        y_pred = model.predict(images)
        
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
    print(f"Dice Coefficient mean: {np.mean(dice_results):.5f}, standard deviation: {np.std(dice_results):.5f}")
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
        np.save(f"{filename_base}_Dice.npy", dice_results)
        np.save(f"{filename_base}_Jaccard.npy", jaccard_results)
        np.save(f"{filename_base}_Sensitivity.npy", sensitivity_results)
        np.save(f"{filename_base}_Specificity.npy", specificity_results)
        
    # Cleanup intermediate results to free memory
    del dice_results, jaccard_results, sensitivity_results, specificity_results