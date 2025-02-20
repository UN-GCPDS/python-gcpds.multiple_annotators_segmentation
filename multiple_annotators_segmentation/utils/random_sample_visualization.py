import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def random_sample_visualization(partition_dataset):
    """
    Visualizes a random sample from a given TensorFlow dataset by selecting a random batch and sample.

    Parameters:
        partition_dataset (tf.data.Dataset): A TensorFlow dataset containing batches of images, masks, and annotator IDs.

    Returns:
        None: This function generates a matplotlib plot for visualization but does not return any values.

    Example:
        >>> dataset = ... # Your dataset
        >>> visualization_dataset_random_sample(dataset)
    """

    # Select a random batch index
    batch = random.randint(0, len(partition_dataset)-1)

    # Skip to the desired batch and take one batch
    for images, masks, id_annotator, orig_masks in partition_dataset.skip(batch).take(1):
        print(f"Images in the batch: {images.shape}, Masks in the batch: {masks.shape}, ID annotator in the batch: {id_annotator.shape}, Original masks in the batch: {orig_masks.shape}")
    

    # Select a random sample within the batch
    sample = random.randint(0, images.shape[0]-1)

    # Permute the axes
    permuted_masks = tf.transpose(masks, perm=[0, 2, 3, 1]) # NCHW --> NHWC

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    # Display the original image for the selected sample
    axes[0,0].set_title('Image')
    axes[0,0].imshow(images[sample])  # Show the sample image
    axes[0,0].axis('off')  # Hide the axes for a cleaner look

    axes[0,1].set_title("Annotator's masks of segmentation")
    axes[0,1].imshow(orig_masks[sample])
    axes[0,1].axis('off')

    axes[1,0].set_title(f"Annotator {tf.argmax(id_annotator[sample]).numpy()}'s masks of segmentation for class 0")
    axes[1,0].imshow(permuted_masks[sample, :, :, 0])
    axes[1,0].axis('off') 

    axes[1,1].set_title(f"Annotator {tf.argmax(id_annotator[sample]).numpy()}'s masks of segmentation for class 1")
    axes[1,1].imshow(permuted_masks[sample, :, :, 1])
    axes[1,1].axis('off')

    plt.show()