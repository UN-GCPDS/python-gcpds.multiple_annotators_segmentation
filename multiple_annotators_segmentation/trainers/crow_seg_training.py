import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from ..losses import noisy_label_loss_tf
from ..metrics import dice_metric, jaccard_metric, sensitivity_metric, specificity_metric



class Crow_Seg_Training:
    """
    Class for managing the training of a segmentation model with custom loss functions and dual optimizers.

    This class handles the training and validation loops for a segmentation model. It supports noisy label loss
    calculation, dual optimizer training (for separate model components), and optional Weights & Biases (WandB)
    monitoring.

    Parameters
    ----------
    model : tf.keras.Model
        The segmentation model to train. This model should return both the segmentation prediction and confidence maps
        (CMS) when called.
    train_dataset : tf.data.Dataset
        Dataset providing training batches. Each batch should contain (images, masks, annotator_ids) and optionally
        ground truth masks if `all_ground_truths=True`.
    valid_dataset : tf.data.Dataset
        Dataset providing validation batches. Similar to the train dataset, it should include data for validation.
    epochs : int, optional, default=60
        Number of epochs to train the model.
    wandb_monitoring : list or None, optional, default=None
        Configuration for WandB monitoring. Must be a list with exactly three strings: [API_KEY, PROJECT_NAME, RUN_NAME].
        Set to None to disable monitoring.
    all_ground_truths : bool, optional, default=False
        If True, the datasets include ground truth masks, and metrics like DICE, Jaccard, sensitivity, and specificity
        will be calculated and tracked.

    Methods
    -------
    wandb_logging() :
        Initializes Weights & Biases logging if configured.

    calculated_metrics(y_true, y_pred) :
        Computes segmentation metrics (DICE, Jaccard, sensitivity, specificity) while handling NaN values.

    train_step(images, masks, ann_ids, orig_mask=None) :
        Performs a single training step, calculates loss, updates model weights with dual optimizers, and returns
        metrics or loss.

    val_step(images, masks, ann_ids, orig_mask=None) :
        Performs a single validation step, calculates loss and metrics without updating model weights.

    training() :
        Executes the full training loop over all epochs, including validation and optional model checkpointing.

    start() :
        Initiates the training process.

    Notes
    -----
    - The model must have components named according to the patterns in `image_cm_patterns` to separate optimizers.
    - NaN values in metrics or loss are replaced with 0 to avoid silent failures.
    - The best model is saved based on either validation DICE score (if `all_ground_truths=True`) or validation loss.
    - Weights & Biases artifact logging is performed at the end of training if monitoring is enabled.

    Examples
    --------
    >>> from crowlib.models import ExampleSegmentationModel
    >>> # Initialize model and datasets
    >>> model = ExampleSegmentationModel(num_classes=2)
    >>> train_data = ...  # Prepare your training dataset
    >>> valid_data = ...  # Prepare your validation dataset
    >>> # Configure WandB monitoring
    >>> wandb_config = ['your_api_key', 'your_project', 'your_run_name']
    >>> # Initialize the training manager
    >>> trainer = Crow_Seg_Training(model, train_data, valid_data, epochs=20, wandb_monitoring=wandb_config, all_ground_truths=True)
    >>> # Start training
    >>> trainer.start()

    """

    def __init__(self, model, train_dataset, valid_dataset, epochs=60, wandb_monitoring=None, all_ground_truths=False):
        """
        Initialize the Crow_Seg_Training class with the given parameters.

        Parameters
        ----------
        model : tf.keras.Model
            The segmentation model to train. This model should return both the segmentation prediction and confidence maps
            (CMS) when called.
        train_dataset : tf.data.Dataset
            Dataset providing training batches. Each batch should contain (images, masks, annotator_ids) and optionally
            ground truth masks if `all_ground_truths=True`.
        valid_dataset : tf.data.Dataset
            Dataset providing validation batches. Similar to the train dataset, it should include data for validation.
        epochs : int, optional, default=60
            Number of epochs to train the model.
        wandb_monitoring : list or None, optional, default=None
            Configuration for WandB monitoring. Must be a list with exactly three strings: [API_KEY, PROJECT_NAME, RUN_NAME].
            Set to None to disable monitoring.
        all_ground_truths : bool, optional, default=False
            If True, the datasets include ground truth masks, and metrics like DICE, Jaccard, sensitivity, and specificity
            will be calculated and tracked.
        """

        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.epochs = epochs
        self.optimizer_image_cm = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.optimizer_resunet = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.min_trace = False
        self.alpha = 1
        self.best_val_dice = 0.0
        self.best_val_loss = float('inf')
        self.wandb_monitoring = wandb_monitoring
        self.all_ground_truths = all_ground_truths
        self.image_cm_patterns = ['image_cm_', 'dense_annotator', 'feature_concat', 'dense_output', 'output_norm']
        self.run = None  # Initialize run here to handle WandB state

    def wandb_logging(self):
        """
        Sets up Weights & Biases monitoring if configured.

        Returns
        -------
        str
            'Tracking in WandB' if setup is successful, 'Normal' otherwise.

        Raises
        ------
        ValueError
            If `wandb_monitoring` is not None or a list of three strings.
        """
        if self.wandb_monitoring is None:
            return 'Normal'
        elif isinstance(self.wandb_monitoring, list) and len(self.wandb_monitoring) == 3 and all(isinstance(item, str) for item in self.wandb_monitoring):
            import wandb  # Ensure wandb is imported here
            wandb.login(key=self.wandb_monitoring[0])
            self.run = wandb.init(project=self.wandb_monitoring[1], name=self.wandb_monitoring[2])
            return 'Tracking in WandB'
        else:
            raise ValueError("The value is not valid. It must be None or a list with exactly three strings.")

    def calculated_metrics(self, y_true, y_pred):
        """
        Computes segmentation metrics while handling NaN values.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth labels.
        y_pred : tf.Tensor
            Predicted labels from the model.

        Returns
        -------
        tuple
            (dice, jaccard, sensitivity, specificity) with NaN values replaced by 0.0.
        """
        dice = dice_metric(y_true, y_pred)
        jaccard = jaccard_metric(y_true, y_pred)
        sensitivity = sensitivity_metric(y_true, y_pred)
        specificity = specificity_metric(y_true, y_pred)

        # Handle NaN values
        dice = tf.where(tf.math.is_nan(dice), 0.0, dice)
        jaccard = tf.where(tf.math.is_nan(jaccard), 0.0, jaccard)
        sensitivity = tf.where(tf.math.is_nan(sensitivity), 0.0, sensitivity)
        specificity = tf.where(tf.math.is_nan(specificity), 0.0, specificity)

        return dice, jaccard, sensitivity, specificity

    def train_step(self, images, masks, ann_ids, orig_mask=None):
        """
        Executes a single training step.

        Parameters
        ----------
        images : tf.Tensor
            Input images.
        masks : tf.Tensor
            Annotator-provided masks.
        ann_ids : tf.Tensor
            Annotator IDs.
        orig_mask : tf.Tensor, optional
            Ground truth masks (if available).

        Returns
        -------
        float or tuple
            If `all_ground_truths=True`, returns (loss, dice, jaccard, sensitivity, specificity).
            Otherwise, returns the training loss.
        """
        masks = tf.argmax(masks, axis=1)

        with tf.GradientTape() as tape:
            y_pred, cms = self.model((images, ann_ids), training=True)

            loss = noisy_label_loss_tf(
                tf.transpose(y_pred, perm=[0, 3, 1, 2]),
                cms,
                masks,
                min_trace=self.min_trace,
                alpha=self.alpha
            )

        if not tf.math.is_finite(loss) or tf.math.is_nan(loss):
            loss = 0.0
            gradients = None  # No gradients to apply
        else:
            # Separate parameters and gradients for Image_CM and ResUNet components
            image_cm_vars = []
            image_cm_grads = []
            resunet_vars = []
            resunet_grads = []

            for var, grad in zip(self.model.trainable_variables, gradients):
                if any(pattern in var.name for pattern in self.image_cm_patterns):
                    image_cm_vars.append(var)
                    image_cm_grads.append(grad)
                else:
                    resunet_vars.append(var)
                    resunet_grads.append(grad)
            
            # Update model weights using separate optimizers
            self.optimizer_image_cm.apply_gradients(zip(image_cm_grads, image_cm_vars))
            self.optimizer_resunet.apply_gradients(zip(resunet_grads, resunet_vars))

        if orig_mask is not None:
            dice, jaccard, sensitivity, specificity = self.calculated_metrics(orig_mask, y_pred[..., 1:2])
            return loss, dice, jaccard, sensitivity, specificity
        else:
            return loss

    def val_step(self, images, masks, ann_ids, orig_mask=None):
        """
        Executes a single validation step.

        Parameters
        ----------
        images : tf.Tensor
            Input images.
        masks : tf.Tensor
            Annotator-provided masks.
        ann_ids : tf.Tensor
            Annotator IDs.
        orig_mask : tf.Tensor, optional
            Ground truth masks (if available).

        Returns
        -------
        float or tuple
            If `all_ground_truths=True`, returns (loss, dice, jaccard, sensitivity, specificity).
            Otherwise, returns the validation loss.
        """
        masks = tf.argmax(masks, axis=1)

        y_pred, cms = self.model((images, ann_ids), training=False)

        loss = noisy_label_loss_tf(
            tf.transpose(y_pred, perm=[0, 3, 1, 2]),
            cms,
            masks,
            min_trace=self.min_trace,
            alpha=self.alpha
        )

        loss = tf.reduce_mean(loss) if not tf.math.is_nan(loss) else 0.0

        if orig_mask is not None:
            dice, jaccard, sensitivity, specificity = self.calculated_metrics(orig_mask, y_pred[..., 1:2])
            return loss, dice, jaccard, sensitivity, specificity
        else:
            return loss

    def training(self):
        """
        Executes the full training loop.

        This method handles the training and validation epochs, updates model weights, tracks metrics, and optionally
        saves the best model and logs data to Weights & Biases.
        """
        type_training = self.wandb_logging()
        self.start_time = time.time()

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            # Update parameters after epoch 5
            if epoch == 5:
                self.min_trace = True
                self.alpha = 0.4
                print(f"Minimize trace activated!, Alpha updated to {self.alpha}")

            total_train_loss = 0.0
            num_train_batches = 0
            total_train_dice = 0.0
            total_train_jaccard = 0.0
            total_train_sensitivity = 0.0
            total_train_specificity = 0.0

            # Training loop
            for data_batch in self.train_dataset:
                data_length = len(data_batch)
                if self.all_ground_truths and data_length == 4:
                    images, masks, ann_ids, orig_mask = data_batch
                else:
                    images, masks, ann_ids = data_batch
                    orig_mask = None

                if self.all_ground_truths:
                    step_result = self.train_step(images, masks, ann_ids, orig_mask)
                    if len(step_result) == 5:
                        batch_loss, batch_dice, batch_jaccard, batch_sensitivity, batch_specificity = step_result
                        total_train_loss += batch_loss.numpy()
                        total_train_dice += batch_dice.numpy() if tf.is_tensor(batch_dice) else batch_dice
                        total_train_jaccard += batch_jaccard.numpy() if tf.is_tensor(batch_jaccard) else batch_jaccard
                        total_train_sensitivity += batch_sensitivity.numpy() if tf.is_tensor(batch_sensitivity) else batch_sensitivity
                        total_train_specificity += batch_specificity.numpy() if tf.is_tensor(batch_specificity) else batch_specificity
                else:
                    batch_loss = self.train_step(images, masks, ann_ids)
                    total_train_loss += batch_loss.numpy()

                num_train_batches += 1

            avg_train_loss = total_train_loss / num_train_batches
            avg_train_dice = total_train_dice / num_train_batches
            avg_train_jaccard = total_train_jaccard / num_train_batches
            avg_train_sensitivity = total_train_sensitivity / num_train_batches
            avg_train_specificity = total_train_specificity / num_train_batches

            # Validation loop
            total_val_loss = 0.0
            num_val_batches = 0
            total_val_dice = 0.0
            total_val_jaccard = 0.0
            total_val_sensitivity = 0.0
            total_val_specificity = 0.0

            for data_batch in self.valid_dataset:
                data_length = len(data_batch)
                if self.all_ground_truths and data_length == 4:
                    images, masks, ann_ids, orig_mask = data_batch
                else:
                    images, masks, ann_ids = data_batch
                    orig_mask = None

                if self.all_ground_truths:
                    step_result = self.val_step(images, masks, ann_ids, orig_mask)
                    if len(step_result) == 5:
                        batch_loss, batch_dice, batch_jaccard, batch_sensitivity, batch_specificity = step_result
                        total_val_loss += batch_loss.numpy()
                        total_val_dice += batch_dice.numpy() if tf.is_tensor(batch_dice) else batch_dice
                        total_val_jaccard += batch_jaccard.numpy() if tf.is_tensor(batch_jaccard) else batch_jaccard
                        total_val_sensitivity += batch_sensitivity.numpy() if tf.is_tensor(batch_sensitivity) else batch_sensitivity
                        total_val_specificity += batch_specificity.numpy() if tf.is_tensor(batch_specificity) else batch_specificity
                else:
                    batch_loss = self.val_step(images, masks, ann_ids)
                    total_val_loss += batch_loss.numpy()

                num_val_batches += 1

            avg_val_loss = total_val_loss / num_val_batches
            avg_val_dice = total_val_dice / num_val_batches
            avg_val_jaccard = total_val_jaccard / num_val_batches
            avg_val_sensitivity = total_val_sensitivity / num_val_batches
            avg_val_specificity = total_val_specificity / num_val_batches

            # Save best model
            if self.all_ground_truths:
                if avg_val_dice > self.best_val_dice:
                    self.best_val_dice = avg_val_dice
                    self.model.save('./models/best_model.keras')
            else:
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.model.save('./models/best_model.keras')

            # Time elapsed
            elapsed_time = time.time() - self.start_time
            elapsed_minutes = int(elapsed_time // 60)
            elapsed_seconds = int(elapsed_time % 60)

            print(f"Training Loss: {'zero' if avg_train_loss == 0 else f'{avg_train_loss:.4f}'} | "
                  f"Validation Loss: {'zero' if avg_val_loss == 0 else f'{avg_val_loss:.4f}'} | "
                  f"Time: {elapsed_minutes}m {elapsed_seconds}s")

            if self.all_ground_truths:
                print(f"Training DICE: {'zero' if avg_train_dice == 0 else f'{avg_train_dice:.4f}'} | "
                      f"Validation DICE: {'zero' if avg_val_dice == 0 else f'{avg_val_dice:.4f}'}")
                if self.run:
                    self.run.log({
                        'Training Loss': avg_train_loss,
                        'Validation Loss': avg_val_loss,
                        'Training DICE': avg_train_dice,
                        'Validation DICE': avg_val_dice,
                        'Training Jaccard': avg_train_jaccard,
                        'Validation Jaccard': avg_val_jaccard,
                        'Training Sensitivity': avg_train_sensitivity,
                        'Validation Sensitivity': avg_val_sensitivity,
                        'Training Specificity': avg_train_specificity,
                        'Validation Specificity': avg_val_specificity
                    })

        # Finish WandB run
        if self.run:
            artifact = wandb.Artifact('best_model', type='model')
            artifact.add_file('./models/best_model.keras')
            self.run.log_artifact(artifact)
            self.run.finish()

        print("Training finished!")

    def start(self):
        """
        Initiates the training process by calling the `training` method.
        """
        self.training()