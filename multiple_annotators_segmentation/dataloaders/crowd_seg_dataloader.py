import os
import re
import glob
import tensorflow as tf

class Crowd_seg_DataLoader:
    """
    A data loader class for crowd segmentation datasets.

    This class handles loading and preprocessing of patch images and their corresponding masks
    for a crowd segmentation task. It supports multiple annotators and classes.

    Attributes:
        data_dir (str): The directory where the data is stored.
        batch_size (int): The size of the batches to be used for training.
        image_size (tuple): The size to which images will be resized.
        num_classes (int): The number of classes in the dataset.
        num_annotators (int): The number of annotators in the dataset.
        partition (str): The partition of the dataset to load (e.g., 'train', 'val', 'test').
    """

    def __init__(self, data_dir, batch_size, image_size, num_classes, num_annotators, partition):
        """
        Initializes the Crowd_seg_DataLoader with the given parameters.

        Args:
            data_dir (str): The directory where the data is stored.
            batch_size (int): The size of the batches to be used for training.
            image_size (tuple): The size to which images will be resized.
            num_classes (int): The number of classes in the dataset.
            num_annotators (int): The number of annotators in the dataset.
            partition (str): The partition of the dataset to load (e.g., 'train', 'val', 'test').
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_annotators = num_annotators
        self.partition = partition

    def load_patch_images(self):
        """
        Loads the patch images from the specified directory and sorts them alphanumerically.

        Returns:
            tf.data.Dataset: A dataset of processed image tensors.
        """
        # Create the path for patch images
        patch_path_pattern = os.path.join(self.data_dir, self.partition, 'patches', '*.png')
        patch_files = glob.glob(patch_path_pattern)

        # Sort patch files alphanumerically
        def alphanumeric_key(s):
            # Split the string into parts (numbers and text)
            parts = re.split(r'(\d+)', s)
            # Convert numeric parts to integers for proper sorting
            return [int(part) if part.isdigit() else part for part in parts]

        patch_files = sorted(patch_files, key=alphanumeric_key)

        self.num_samples = len(patch_files)

        # Print the complete path and the number of found patch images
        print(f"Complete path for patches: {patch_path_pattern}")
        print(f"Number of patch files found: {len(patch_files)}")

        # Create a TensorFlow dataset from the patch file paths
        patch_ds = tf.data.Dataset.from_tensor_slices(patch_files)

        # Map each file path to a processed image
        patch_ds = patch_ds.map(self.process_patch, num_parallel_calls=tf.data.AUTOTUNE)

        return patch_ds

    def load_masks(self):
        """
        Loads the masks for each annotator and class from the specified directory and constructs a dataset.

        Returns:
            tuple: A tuple containing a dataset of processed mask tensors and a dataset of annotator IDs.
        """
        # Assume that mask preparation logic is implemented
        mask_path_main = os.path.join(self.data_dir, self.partition, 'masks')
        masks_path = []
        ids_annotators = []

        for sample in range(self.num_samples):
            masks_sample = []

            # Randomly select an annotator for each sample
            random_annotator = tf.random.uniform(shape=(), minval=0, maxval=self.num_annotators, dtype=tf.int32)

            # Create a one-hot tensor for the selected annotator
            one_hot_tensor = tf.one_hot(random_annotator, depth=self.num_annotators)

            ids_annotators.append(one_hot_tensor)

            for class_id in range(self.num_classes):
                mask_annotation_path = os.path.join(mask_path_main, f'annotator_{random_annotator+1}', f'class_{class_id}', f'sample_{sample}.png')
                masks_sample.append(mask_annotation_path)

            masks_path.append(masks_sample)

        ids_annotators = tf.data.Dataset.from_tensor_slices(ids_annotators)

        # Create dataset from `masks_path` and apply `process_masks` to each set of paths
        masks_ds = tf.data.Dataset.from_tensor_slices(masks_path)
        masks_ds = masks_ds.map(self.process_masks, num_parallel_calls=tf.data.AUTOTUNE)

        return masks_ds, ids_annotators

    def load_orig_masks(self):
        """
        Loads the masks for the ground truth from the specified directory.

        Returns:
            tf.data.Dataset: A dataset of processed mask tensors.
        """

        # Create the path for masks images
        mask_path_pattern = os.path.join(self.data_dir, self.partition, 'masks', 'ground_truth','*.png')
        mask_files = glob.glob(mask_path_pattern)

        # Sort mask files alphanumerically
        def alphanumeric_key(s):
            # Split the string into parts (numbers and text)
            parts = re.split(r'(\d+)', s)
            # Convert numeric parts to integers for proper sorting
            return [int(part) if part.isdigit() else part for part in parts]

        mask_files = sorted(mask_files, key=alphanumeric_key)

        # Print the complete path and the number of found masks images
        print(f"Complete path for masks (ground truth): {mask_path_pattern}")
        print(f"Number of ground truth mask files found: {len(mask_files)}")

        # Create a TensorFlow dataset from the masks file paths
        masks_ds = tf.data.Dataset.from_tensor_slices(mask_files)

        # Map each file mask to a processed image
        masks_ds = masks_ds.map(self.process_orig_masks, num_parallel_calls=tf.data.AUTOTUNE)

        return masks_ds

    def process_patch(self, file_path):
        """
        Reads and processes a patch image for standardization.

        Args:
            file_path (str): Path to the image file.

        Returns:
            tf.Tensor: Normalized image tensor.
        """
        img = tf.io.read_file(file_path)  # Read the image file
        img = tf.image.decode_jpeg(img, channels=3)  # Decode JPEG image
        img = tf.image.resize(img, self.image_size)  # Resize image
        img = tf.cast(img, tf.float32)  # Convert to float32
        img /= 255.0  # Normalize to [0, 1]
        return img

    def process_masks(self, sample_paths):
        """
        Processes multiple mask images, resizes, and normalizes them.

        Args:
            sample_paths: Tensor containing paths to mask images.

        Returns:
            tf.Tensor: Processed mask tensor with shape [classes, height, width].
        """
        # Decode and process images
        decoded_images = tf.map_fn(
            tf.io.read_file,
            sample_paths,
            dtype=tf.string,
            parallel_iterations=4
        )

        masks = tf.map_fn(
            lambda x: tf.io.decode_png(x, channels=1),
            decoded_images,
            dtype=tf.uint8,
            parallel_iterations=4
        )

        # Resize and normalize
        masks = tf.map_fn(
            lambda x: tf.image.resize(x, size=self.image_size),
            masks,
            dtype=tf.float32
        )

        masks = tf.cast(masks, tf.float32)  # Convert to float32
        masks = masks / 255.0  # Normalize to [0, 1]
        masks = tf.squeeze(masks, axis=-1)  # Remove the last dimension

        return masks

    def process_orig_masks(self, file_mask):
        """
        Processes a ground truth mask image for consistency with other datasets.

        Args:
            file_mask (str): Path to the mask image file.

        Returns:
            tf.Tensor: Processed mask tensor.
        """
        mask = tf.io.read_file(file_mask)  # Read the image file
        mask = tf.image.decode_jpeg(mask, channels=1)  # Decode JPEG image
        mask = tf.image.resize(mask, self.image_size)  # Resize image
        mask = tf.cast(mask, tf.float32)  # Convert to float32
        mask /= 255.0  # Normalize to [0, 1]
        return mask

    def get_dataset(self):
        """
        Combines patch images, masks, and annotator IDs into a single dataset.

        Returns:
            tf.data.Dataset: A dataset containing batches of patch images, masks, annotator IDs, and original masks.
        """
        patch_ds = self.load_patch_images()
        masks_ds, id_ds = self.load_masks()
        masks_ds_orig = self.load_orig_masks()

        # Combine all elements into a single dataset
        dataset = tf.data.Dataset.zip((patch_ds, masks_ds, id_ds, masks_ds_orig))

        # Apply batching and prefetching for optimization
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset