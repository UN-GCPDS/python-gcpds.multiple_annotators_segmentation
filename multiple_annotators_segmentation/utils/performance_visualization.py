import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def performance_visualization(partition_dataset, model, config, original_masks=True):
    """
    Visualize model performance by displaying original image, annotator masks, and model predictions.
    
    This function randomly selects a batch and sample from the dataset, then creates a
    visualization grid showing:
    1. The original input image
    2. The ground truth masks from the original dataset (if available)
    3. Sample annotations from different annotators for different classes
    4. The model's predicted segmentation masks for different classes
    
    The visualization helps in qualitative assessment of the model's performance and
    understanding the variability in annotations across different annotators.
    
    Args:
        partition_dataset: TensorFlow dataset containing images and masks (and optionally original masks)
        model: Trained segmentation model to generate predictions
        config: Dictionary containing model configuration parameters, must include:
               - 'Num of annotators': Number of different annotators in the dataset
               - 'Number of classes': Number of segmentation classes
        original_masks: Boolean flag indicating if the dataset includes original ground truth
                       masks (default: True)
        
    Returns:
        None: Displays the visualization using matplotlib
    """
    
    # Select a random batch index
    batch = random.randint(0, len(partition_dataset)-1)
    
    # Skip to the desired batch and take one batch
    if original_masks:
        for images, masks, orig_masks in partition_dataset.skip(batch).take(1):
            print(f"Images in the batch: {images.shape}, Masks in the batch: {masks.shape}, Original masks in the batch: {orig_masks.shape}")
    else:
        for images, masks in partition_dataset.skip(batch).take(1):
            print(f"Images in the batch: {images.shape}, Masks in the batch: {masks.shape}")
    
    # Select a random sample within the batch
    sample = random.randint(0, images.shape[0]-1)
    
    # Generate predictions for the selected sample
    y_pred = model.predict(images[sample:sample+1])
    
    # Determine number of columns for visualization based on available masks or classes
    columns = masks.shape[-1] if masks.shape[-1] <= 5 else 5
    
    if original_masks:
        # Adjust columns to accommodate original masks if needed
        classes_orig_mask = orig_masks.shape[-1] if orig_masks.shape[-1] <= 5 else 5
        columns = max(classes_orig_mask, columns)
        
    # Determine number of rows needed for the visualization
    rows = 4 if original_masks else 3
    
    # Create figure with GridSpec for more control over subplot arrangement
    fig = plt.figure(figsize=(10, 6))
    # Add an extra row for titles, with height ratios for better spacing
    gs = fig.add_gridspec(rows + 1, columns, height_ratios=[3] * rows + [0.1])
    
    # Create axes manually for each subplot position
    axes = [[fig.add_subplot(gs[r, c]) for c in range(columns)] for r in range(rows)]
    
    # Add title axes for annotator and prediction sections
    title_ax = fig.add_subplot(gs[rows-2, :])
    title_ax.set_title("Annotations of different annotators and classes", loc='left')
    title_ax.axis('off')  # Hide axes for cleaner look
    
    title_ax = fig.add_subplot(gs[rows-1, :])
    title_ax.set_title("Masks predicted for different classes", loc='left')
    title_ax.axis('off')  # Hide axes for cleaner look
    
    # Display the original image for the selected sample
    axes[0][0].set_title('Image')
    axes[0][0].imshow(images[sample])  # Show the selected sample image
    
    if original_masks:
        # Display original ground truth masks if available
        for i in range(columns):
            # Check if the class index is within the range of available masks
            if i < orig_masks.shape[-1] and orig_masks[sample,:,:,i:i+1].shape[-1] == 1:
                # Adjust title based on number of classes
                axes[1][i].set_title(
                    f"Single original mask of segmentation" if orig_masks.shape[-1] == 1 
                    else f"Original masks of segmentation for class {i}",
                    loc='left'
                )
                axes[1][i].imshow(orig_masks[sample,:,:,i:i+1])
            axes[1][i].axis('off')
    
    # Generate random indices for annotators and classes to display (limited to 5)
    # Read number of annotators from config dictionary
    annotator_list = random.sample(range(config['Num of annotators']), min(5, config['Num of annotators'])) \
        if (isinstance(config['Num of annotators'],int) and config['Num of annotators'] >= 2) else [0]*columns
    
    # Read number of classes from config dictionary
    classes_list = random.sample(range(config['Number of classes']), min(5, config['Number of classes'])) \
        if (isinstance(config['Number of classes'],int) and config['Number of classes'] >= 2) else [0]*columns
    
    # Display different annotator masks for different classes
    for i in range(columns):
        try:
            # Calculate the index in the masks tensor based on annotator and class
            # The formula maps from (annotator, class) to the corresponding channel in the masks tensor
            # Format: masks[batch_idx, height, width, annotator_idx + class_idx * num_annotators]
            axes[rows-2][i].imshow(
                masks[sample, :, :, annotator_list[i]+classes_list[i]*config['Num of annotators']:
                              annotator_list[i]+classes_list[i]*config['Num of annotators']+1]
            )  
        except IndexError:
            # Skip if the index is out of range (can happen with random sampling)
            pass
    
    # Display model predictions for different classes
    for i in range(columns):
        try:
            # Apply threshold of 0.5 to convert prediction probabilities to binary masks
            # This creates a binary segmentation mask from the model's continuous predictions
            axes[rows-1][i].imshow(
                tf.where(y_pred[0, :, :, classes_list[i]:classes_list[i]+1] >= 0.5, 1.0, 0.0)
            )
        except IndexError:
            # Skip if the class index is out of range
            pass
        
        # Hide axes for cleaner visualization
        axes[0][i].axis('off')
        axes[rows-2][i].axis('off')
        axes[rows-1][i].axis('off')
    
    # Adjust layout for better spacing
    fig.tight_layout()
    plt.show()