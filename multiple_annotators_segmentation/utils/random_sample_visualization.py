import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def random_sample_visualization(partition_dataset, original_masks=True):
    """
   Visualizes a random sample from a dataset containing images, segmentation masks, and annotator IDs.
   
   This function selects a random batch from the provided dataset, then selects a random sample 
   within that batch and creates a visualization. The visualization shows the original image, 
   the annotator's segmentation masks for different classes, and optionally the original ground 
   truth masks when available.
   
   Args:
       partition_dataset: A TensorFlow dataset containing batches of (images, masks, annotator_ids)
                         or (images, masks, annotator_ids, original_masks) if original_masks=True
       original_masks (bool, optional): Whether the dataset includes original ground truth masks.
                                       Defaults to True.
   
   Returns:
       None. Displays a matplotlib figure showing:
           - The original image
           - The original ground truth segmentation masks (if original_masks=True)
           - The annotator's segmentation masks for each class
           
   Notes:
       - The function prints shape information about the selected batch
       - The visualization layout adapts based on the number of segmentation classes
       - Maximum 5 classes are displayed (columns) if there are more
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

    columns = masks.shape[1] if masks.shape[1] <= 5 else 5

    if original_masks:
        
        classes_orig_mask = orig_masks.shape[-1] if orig_masks.shape[-1] <= 5 else 5
        columns = max(classes_orig_mask, columns)
        
    rows = 3 if original_masks else 2
    
    # Crear figura con GridSpec para tener más control
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(rows + 1, columns, height_ratios=[3] * rows + [0.1])  # Agregar una fila extra para el título
    
    # Crear axes manualmente
    axes = [[fig.add_subplot(gs[r, c]) for c in range(columns)] for r in range(rows)]
    
    # Agregar un título para la última fila en la fila adicional
    title_ax = fig.add_subplot(gs[rows-1, :])
    title_ax.set_title(f"Annotator {tf.argmax(id_annotator[sample]).numpy()+1}s' masks of segmentation for different classes",loc='left')
    title_ax.axis('off')  # Ocultar ejes
    
    # Display the original image for the selected sample
    axes[0][0].set_title('Image')
    axes[0][0].imshow(images[sample])  # Show the sample image

    if original_masks:
        for i in range(columns):
            if i < orig_masks.shape[-1] and orig_masks[sample,:,:,i:i+1].shape[-1] == 1:
                axes[1][i].set_title(f"Single original mask of segmentation" if orig_masks.shape[-1] == 1 else f"Original masks of segmentation for class {i}",loc='left')
                axes[1][i].imshow(orig_masks[sample,:,:,i:i+1])
            axes[1][i].axis('off')

    classes_list = [random.randint(0, permuted_masks.shape[-1] - 1) for _ in range(columns)] if (isinstance(permuted_masks.shape[-1],int) and permuted_masks.shape[-1] >= 2) else [0]*columns

    for i in range(columns):
        try:
            axes[2][i].imshow(permuted_masks[sample, :, :, classes_list[i]:classes_list[i]+1])
        except:
            pass

        axes[0,i].axis('off')  # Hide the axes for a cleaner look
        axes[rows-1,i].axis('off')

    fig.tight_layout() 
    plt.show()