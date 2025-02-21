import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def random_sample_visualization(partition_dataset, original_masks=True):
    """
    Visualizes a random sample from a batched TensorFlow dataset.

    This function selects a random batch and sample from the provided dataset,
    then displays the original image, segmentation masks, and (optionally) ground truth masks.

    Parameters:
        partition_dataset (tf.data.Dataset): A batched TensorFlow dataset containing tuples of:
            - Images: Batch of input images.
            - Masks: Batch of segmentation masks (shape: [batch_size, num_classes, height, width]).
            - Annotator IDs: Batch of one-hot encoded annotator identifiers.
            - Original masks (optional): Batch of ground truth masks (if `original_masks=True`).
        original_masks (bool, optional): Whether to include ground truth masks in the visualization. Defaults to True.

    Returns:
        None: This function generates a matplotlib plot for visualization but does not return any values.

    Example:
        >>> dataset = ...  # Your batched dataset
        >>> random_sample_visualization(dataset, original_masks=True)

    Note:
        - The input dataset must be batched using `dataset.batch()` before calling this function.
        - The dataset elements must contain the appropriate number of components based on `original_masks`:
            - 3 components (images, masks, annotator IDs) if `original_masks=False`.
            - 4 components (images, masks, annotator IDs, original masks) if `original_masks=True`.
        - The masks are expected to be in NCHW format (batch, classes, height, width), which is permuted to NHWC for visualization.
        - The annotator IDs are assumed to be one-hot encoded tensors.

    Visualization Components:
        - **Top-Left**: Original input image.
        - **Top-Right**: Ground truth mask (if `original_masks=True`).
        - **Bottom-Left**: Segmentation mask for class 0 from the selected annotator.
        - **Bottom-Right**: Segmentation mask for class 1 from the selected annotator.

    Raises:
        ValueError: If the dataset elements do not match the expected structure based on `original_masks`.

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
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    # Display the original image for the selected sample
    axes[0,0].set_title('Image')
    axes[0,0].imshow(images[sample])  # Show the sample image
    axes[0,0].axis('off')  # Hide the axes for a cleaner look

    if original_masks:
        axes[0,1].set_title("Original masks of segmentation")
        axes[0,1].imshow(orig_masks[sample])
    axes[0,1].axis('off')

    axes[1,0].set_title(f"Annotator {tf.argmax(id_annotator[sample]).numpy()}'s masks of segmentation for class 0")
    axes[1,0].imshow(permuted_masks[sample, :, :, 0])
    axes[1,0].axis('off') 
    
    axes[1,1].set_title(f"Annotator {tf.argmax(id_annotator[sample]).numpy()}'s masks of segmentation for class 1")
    axes[1,1].imshow(permuted_masks[sample, :, :, 1])
    axes[1,1].axis('off')

    plt.show()