import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def haphazard_sample_visualization(partition_dataset, num_classes, num_annotators, original_masks=True):
    """
    Visualizes a random sample from a segmentation dataset with multiple annotators.
    
    This function selects a random batch and sample from the provided dataset and creates
    a visualization grid showing the original image, ground truth masks (if available),
    and a selection of annotator masks for different classes.
    
    Args:
        partition_dataset (tf.data.Dataset): A TensorFlow dataset containing batches of 
            images and masks, possibly with original ground truth masks.
        num_classes (int): The number of segmentation classes in the dataset.
        num_annotators (int): The number of annotators who provided masks.
        original_masks (bool, optional): Whether the dataset includes original ground truth 
            masks. Defaults to True.
    
    Returns:
        None: The function displays the visualization using matplotlib and does not return a value.
    
    Note:
        The function expects the dataset to yield tuples of (images, masks) or 
        (images, masks, orig_masks) if original_masks is True.
    """
    # Select a random batch index
    batch = random.randint(0, len(partition_dataset)-1)
    
    # Skip to the desired batch and take one batch
    if original_masks:
        # Handle datasets that include ground truth masks
        for images, masks, orig_masks in partition_dataset.skip(batch).take(1):
            print(f"Images in the batch: {images.shape}, Masks in the batch: {masks.shape}, Original masks in the batch: {orig_masks.shape}")
    else:
        # Handle datasets without ground truth masks
        for images, masks in partition_dataset.skip(batch).take(1):
            print(f"Images in the batch: {images.shape}, Masks in the batch: {masks.shape}")
    
    # Select a random sample within the batch
    sample = random.randint(0, images.shape[0]-1)
    
    # Determine the number of columns for the visualization grid
    # Limit to 5 columns maximum to keep the visualization manageable
    columns = masks.shape[-1] if masks.shape[-1] <= 5 else 5
    
    if original_masks:
        # Adjust columns based on the number of classes in original masks
        classes_orig_mask = orig_masks.shape[-1] if orig_masks.shape[-1] <= 5 else 5
        columns = max(classes_orig_mask, columns)
        
    # Determine the number of rows for the visualization
    # 3 rows if original masks are included, 2 rows otherwise
    rows = 3 if original_masks else 2
    
    # Create figure with GridSpec for more layout control
    fig = plt.figure(figsize=(10, 6))
    # Add extra row for the title with appropriate height ratios
    gs = fig.add_gridspec(rows + 1, columns, height_ratios=[3] * rows + [0.1])
    
    # Create axes manually for each cell in the grid
    axes = [[fig.add_subplot(gs[r, c]) for c in range(columns)] for r in range(rows)]
    
    # Add a title for the annotations row in the additional row
    title_ax = fig.add_subplot(gs[rows-1, :])
    title_ax.set_title("Annotations of different annotators and classes", loc='left')
    title_ax.axis('off')  # Hide axes for cleaner look
    
    # Display the original image for the selected sample
    axes[0][0].set_title('Image')
    axes[0][0].imshow(images[sample])  # Show the sample image
    
    if original_masks:
        # Display original ground truth masks if available
        for i in range(columns):
            if i < orig_masks.shape[-1] and orig_masks[sample,:,:,i:i+1].shape[-1] == 1:
                # Set appropriate title based on number of classes
                axes[1][i].set_title(
                    f"single original mask of segmentation" if orig_masks.shape[-1] == 1 
                    else f"Original masks of segmentation for class {i}", 
                    loc='left'
                )
                axes[1][i].imshow(orig_masks[sample,:,:,i:i+1])
            axes[1][i].axis('off')
    
    # Randomly select annotators and classes to display
    # If we have multiple annotators, select up to 5 random ones, otherwise use default
    annotator_list = (
        random.sample(range(num_annotators), min(5, num_annotators)) 
        if (isinstance(num_annotators, int) and num_annotators >= 2) 
        else [0] * columns
    )
    
    # If we have multiple classes, select random classes for each column, otherwise use default
    # Note: There's a typo in the original code with "num*classes" that should be fixed
    classes_list = (
        [random.randint(0, num_classes - 1) for _ in range(columns)] 
        if (isinstance(num_classes, int) and num_classes >= 2) 
        else [0] * columns
    )
    
    # Display mask for each selected annotator and class combination
    for i in range(columns):
        try:
            # Calculate index based on annotator and class
            # Format: annotator_index + class_index * num_annotators
            mask_index = annotator_list[i] + classes_list[i] * num_annotators
            axes[rows-1][i].imshow(
                masks[sample, :, :, mask_index:mask_index+1]
            )  
        except IndexError:
            # Skip if index is out of bounds
            pass
        
        # Hide axes for cleaner visualization
        axes[0][i].axis('off')
        axes[rows-1][i].axis('off')
    
    # Adjust layout for better spacing
    fig.tight_layout()
    
    # Display the visualization
    plt.show()