import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def sample_performance_visualization(partition_dataset, model, original_masks=True):  
    """
    Visualizes model prediction performance on a random sample from a dataset.
    
    This function selects a random batch from the provided dataset, then selects a random sample 
    within that batch and creates a multi-panel visualization. The visualization shows the original 
    image, the annotator's segmentation masks, the model's predicted segmentation masks, and 
    optionally the original ground truth masks when available.
    
    Args:
       partition_dataset: A TensorFlow dataset containing batches of (images, masks, annotator_ids)
                         or (images, masks, annotator_ids, original_masks) if original_masks=True
       model: The trained segmentation model to generate predictions
       original_masks (bool, optional): Whether the dataset includes original ground truth masks.
                                       Defaults to True.
    
    Returns:
       None. Displays a matplotlib figure showing:
           - The original image
           - The original ground truth segmentation masks (if original_masks=True)
           - The annotator's segmentation masks for each class
           - The model's predicted segmentation masks for each class
           
    Notes:
       - The function prints shape information about the selected batch
       - The visualization layout adapts based on the number of segmentation classes
       - Maximum 10 classes are displayed (columns) if there are more
       - Uses tight_layout for better visualization spacing
    """
    # Select a random batch index
    batch = random.randint(0, len(partition_dataset)-1)

    # Skip to the desired batch and take one batch
    if original_masks:
        for images, masks, id_annotator, orig_masks in partition_dataset.skip(batch).take(1):
            print(f"Images in the batch: {images.shape}, Masks in the batch: {masks.shape}, ID annotator in the batch: {id_annotator.shape}, Original masks in the batch: {orig_masks.shape}")
    else:
        for images, masks, id_annotator in partition_dataset.skip(batch).take(1):
            print(f"Images in the batch: {images.shape}, Masks in the batch: {masks.shape}, ID annotator in the batch: {id_annotator.shape}")
    

    # Select a random sample within the batch
    sample = random.randint(0, images.shape[0]-1)

    # Permute the axes
    permuted_masks = tf.transpose(masks, perm=[0, 2, 3, 1]) # NCHW --> NHWC

    y_pred, _ = model.predict((images[sample:sample+1,...],id_annotator[sample:sample+1,...]))

    columns = masks.shape[1] if masks.shape[1] <= 10 else 10

    if original_masks:
        
        classes_orig_mask = orig_masks.shape[-1] if orig_masks.shape[-1] <= 10 else 10

        columns = max(classes_orig_mask, columns)

    rows = 4 if original_masks else 3
    
    fig, axes = plt.subplots(rows, columns, figsize=(10, 6))

    # Display the original image for the selected sample
    axes[0,0].set_title('Image')
    axes[0,0].imshow(images[sample])  # Show the sample image

    if original_masks:
        for i in range(columns):
            if orig_masks[sample,:,:,i:i+1].shape[-1] == 1:
                axes[1,i].set_title(f"Single original mask of segmentation" if orig_masks.shape[-1] == 1 else f"Original masks of segmentation for class {i}")
                axes[1,i].imshow(orig_masks[sample,:,:,i:i+1])
            axes[1,i].axis('off')
            
    for i in range(columns):
        if permuted_masks[sample,:,:,i:i+1].shape[-1] == 1:
            axes[rows-2,i].set_title(f"Annotator {tf.argmax(id_annotator[sample]).numpy()}'s masks of segmentation for class {i}")
            axes[rows-2,i].imshow(permuted_masks[sample, :, :, i:i+1])
        if y_pred[0,:,:,i:i+1].shape[-1] == 1:
            axes[rows-1,i].set_title(f"Predicted masks of segmentation for class {i}")
            axes[rows-1,i].imshow(tf.where(y_pred[0, :, :, i:i+1] > 0.5, 1.0, 0.0))
        axes[0,i].axis('off')  # Hide the axes for a cleaner look
        axes[rows-2,i].axis('off')
        axes[rows-1,i].axis('off')       
        
    fig.tight_layout()   
    plt.show()