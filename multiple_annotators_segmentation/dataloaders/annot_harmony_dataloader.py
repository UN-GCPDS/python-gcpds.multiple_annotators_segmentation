import os
import re
import glob
import tensorflow as tf

class Annot_Harmony_DataLoader:
    """
    A data loader class for handling multiple annotator segmentation datasets.
    
    This class is designed to load and process image patches and their corresponding
    segmentation masks from multiple annotators. It supports loading ground truth masks
    when available and can handle different partitions of the dataset (training, validation, testing).
    
    Attributes:
        data_dir (str): Root directory containing the dataset.
        batch_size (int): Number of samples per batch.
        image_size (tuple): Target size for images and masks (height, width).
        num_classes (int): Number of segmentation classes.
        num_annotators (int): Number of annotators who provided masks.
        partition (str): Dataset partition ('Train', 'Val', 'Test').
        group_all_samples_ground_truth (bool): Whether to include ground truth masks.
        target_class (int, optional): Specific class to focus on, if any.
        num_samples (int): Total number of samples in the dataset.
        existing_classes (list): List of class IDs that have ground truth masks.
    
    The expected directory structure is:
        data_dir/
        ├── partition/
        │   ├── patches/             # Contains all image patches
        │   │   └── *.png
        │   └── masks/
        │       ├── annotator_1/     # Masks from first annotator
        │       │   ├── class_0/
        │       │   │   └── *.png
        │       │   ├── class_1/
        │       │   │   └── *.png
        │       │   └── ...
        │       ├── annotator_2/
        │       │   └── ...
        │       └── ground_truth/    # Ground truth masks when available
        │           ├── class_0/
        │           │   └── *.png
        │           └── ...
    """

    def __init__(self, data_dir, batch_size, image_size, num_classes, num_annotators, partition, all_samples_ground_truth=True, target_class=None):
        """
        Initialize the data loader with dataset parameters.
        
        Args:
            data_dir (str): Root directory containing the dataset.
            batch_size (int): Number of samples per batch.
            image_size (tuple): Target size for images and masks (height, width).
            num_classes (int): Number of segmentation classes.
            num_annotators (int): Number of annotators who provided masks.
            partition (str): Dataset partition ('Train', 'Val', 'Test').
            all_samples_ground_truth (bool): Whether to include ground truth masks. Defaults to True.
            target_class (int, optional): Specific class to focus on. Defaults to None.
        """
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_annotators = num_annotators
        self.partition = partition
        self.group_all_samples_ground_truth = all_samples_ground_truth
        self.target_class = target_class

    def load_patch_images(self):
        """
        Load image patches from the dataset.
        
        Reads all patch images from the specified partition, sorts them alphanumerically,
        and creates a TensorFlow dataset. Each image is processed to have consistent size
        and normalization.
        
        Returns:
            tf.data.Dataset: Dataset of processed image patches.
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
        Load annotator masks from the dataset.
        
        For each annotator and each class, this method finds and loads the corresponding
        segmentation masks. It organizes the masks by sample, class, and annotator.
        
        Returns:
            tf.data.Dataset: Dataset of processed annotation masks.
        """

        # Initialize a list to hold the masks for each annotator
        for annotator in range(self.num_annotators):
            list_annotations = []
            for class_id in range(self.num_classes):
                mask_path_pattern = os.path.join(self.data_dir, self.partition, 'masks', f'annotator_{annotator + 1}', f'class_{class_id}', '*.png')
                found_masks = glob.glob(mask_path_pattern)
                found_masks = sorted(found_masks)
                list_annotations.append(found_masks)

                print(f"Mask path for annotator {annotator + 1}, class {class_id}: {mask_path_pattern}")
                print(f"Number of masks found: {len(found_masks)}")

        masks_path = []
        mask_path_main = os.path.join(self.data_dir, self.partition, 'masks')
        
        # Organize masks by sample, annotator, and class
        for sample in range(self.num_samples):
            masks_sample = []
            for class_id in range(self.num_classes):
                for annotator in range(self.num_annotators):
                    mask_annotation_path = os.path.join(mask_path_main, f'annotator_{annotator+1}', f'class_{class_id}', f'sample_{sample}.png')
                    masks_sample.append(mask_annotation_path)
                    found_masks = glob.glob(mask_annotation_path)
            masks_path.append(masks_sample)
        
        # Create dataset from `masks_path` and apply `process_masks` to each set of paths
        masks_ds = tf.data.Dataset.from_tensor_slices(masks_path)
        masks_ds = masks_ds.map(self.process_masks, num_parallel_calls=tf.data.AUTOTUNE)

        return masks_ds

    def load_orig_masks(self):
        """
        Load ground truth masks from the dataset.
        
        This method loads the original (ground truth) masks for each class and sample.
        It tracks which classes have ground truth masks available and organizes them by sample.
        
        Returns:
            tf.data.Dataset: Dataset of processed ground truth masks.
        """

        mask_path_main = os.path.join(self.data_dir, self.partition, 'masks', 'ground_truth')
        self.existing_classes = []
        
        # Check which classes have ground truth masks
        for class_id in range(self.num_classes):
            masks_sample = []
            mask_path_pattern = os.path.join(mask_path_main, f'class_{class_id}', '*.png')
            found_masks = glob.glob(mask_path_pattern)
            if len(found_masks) != 0:
                self.existing_classes.append(class_id)
            print(f"Original masks path, class {class_id}: {mask_path_pattern}")
            print(f"Number of masks found: {len(found_masks)}")

        masks_path = []
        # Organize ground truth masks by sample and class
        for sample in range(self.num_samples):
            masks_sample = []
            for class_id in self.existing_classes:
                mask_annotation_path = os.path.join(mask_path_main, f'class_{class_id}', f'sample_{sample}.png')
                masks_sample.append(mask_annotation_path)
            masks_path.append(masks_sample)
        
        # Create dataset from `masks_path` and apply `process_masks` to each set of paths
        masks_ds = tf.data.Dataset.from_tensor_slices(masks_path)
        masks_ds = masks_ds.map(self.process_orig_masks, num_parallel_calls=tf.data.AUTOTUNE)

        return masks_ds

    def process_patch(self, file_path):
        """
        Process a single image patch.
        
        Reads, decodes, resizes, and normalizes an image from its file path.
        
        Args:
            file_path (tf.Tensor): Path to the image file.
            
        Returns:
            tf.Tensor: Processed image tensor with shape [height, width, 3].
        """
        
        img = tf.io.read_file(file_path)  # Read the image file
        img = tf.image.decode_jpeg(img, channels=3)  # Decode JPEG image
        img = tf.image.resize(img, self.image_size)  # Resize image
        img = tf.cast(img, tf.float32)  # Convert to float32
        img /= 255.0  # Normalize to [0, 1]
        return img

    def process_masks(self, sample_paths):
        """
        Process a batch of annotation masks.
        
        Reads, decodes, resizes, and normalizes a batch of mask images.
        Organizes masks in the format expected by the model.
        
        Args:
            sample_paths (tf.Tensor): Tensor of mask file paths.
            
        Returns:
            tf.Tensor: Processed mask tensor with shape [height, width, num_classes*num_annotators].
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
        masks = tf.transpose(masks, perm=[1, 2, 0])  # Transpose to [height, width, classes]

        return masks

    def process_orig_masks(self, sample_paths):
        """
        Process a batch of ground truth masks.
        
        Similar to process_masks, but specifically for ground truth masks.
        Reads, decodes, resizes, and normalizes mask images for the ground truth.
        
        Args:
            sample_paths (tf.Tensor): Tensor of ground truth mask file paths.
            
        Returns:
            tf.Tensor: Processed ground truth mask tensor with shape [height, width, num_classes].
        """
        
        # Convert string paths to a list of decoded images using tf.map_fn
        decoded_images = tf.map_fn(
            tf.io.read_file,
            sample_paths,
            dtype=tf.string,
            parallel_iterations=4
        )

        # Decode images
        masks = tf.map_fn(
            lambda x: tf.io.decode_png(x, channels=1),
            decoded_images,
            dtype=tf.uint8,
            parallel_iterations=4
        )
        
        # Resize the masks to a common size
        masks = tf.map_fn(
            lambda x: tf.image.resize(x, size=self.image_size),
            masks,
            dtype=tf.float32
        )

        # Convert to float32 and normalize
        masks = tf.cast(masks, tf.float32) / 255.0
        
        masks = tf.squeeze(masks, axis=-1)  # Remove the last dimension
        masks = tf.transpose(masks, perm=[1, 2, 0])  # Transpose to [height, width, classes]
        
        return masks

    def get_dataset(self):
        """
        Combine patches and masks into a unified dataset.
        
        Loads patches and masks, combines them into a single dataset,
        and applies batching and prefetching for performance optimization.
        Different dataset compositions are created based on the partition type
        and whether ground truth is requested.
        
        Returns:
            tf.data.Dataset: The final dataset with both images and masks.
        """
        
        patch_ds = self.load_patch_images()
        masks_ds = self.load_masks()
        if self.partition in ('Train', 'Val', 'Valid', 'Validation') and self.group_all_samples_ground_truth:
            # For training/validation with ground truth
            masks_ds_orig = self.load_orig_masks()
            dataset = tf.data.Dataset.zip((patch_ds, masks_ds, masks_ds_orig))
        elif self.partition == 'Test':
            # For testing, include ground truth if available
            masks_ds_orig = self.load_orig_masks()
            dataset = tf.data.Dataset.zip((patch_ds, masks_ds, masks_ds_orig))
        else:
            # For other cases, just use patches and annotator masks
            dataset = tf.data.Dataset.zip((patch_ds, masks_ds))
        # Apply batching and prefetching for optimization
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset