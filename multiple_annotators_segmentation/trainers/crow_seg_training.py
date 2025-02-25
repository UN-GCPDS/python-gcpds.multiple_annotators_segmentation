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
    A training class for semantic segmentation models with uncertainty estimation via confusion matrices.
    
    This class manages the training process for segmentation models that incorporate image-level 
    confusion matrices for handling noisy labels. It supports both standard training and validation
    loops with performance tracking, optional Weights & Biases monitoring, and separate optimizers
    for segmentation and confusion matrix components.
    
    Args:
        model: The segmentation model to train, expected to output predictions and confusion matrices
        train_dataset: TensorFlow dataset containing training data batches
        valid_dataset: TensorFlow dataset containing validation data batches
        epochs (int, optional): Number of training epochs. Defaults to 60.
        wandb_monitoring (list, optional): WandB configuration as [API_KEY, PROJECT, RUN_NAME] or None to disable. Defaults to None.
        all_ground_truths (bool, optional): Whether datasets include original ground truth masks. Defaults to False.
        config_model: Configuration dictionary for model parameters, used for WandB logging
    
    Attributes:
        optimizer_image_cm: Optimizer for confusion matrix components (Adam, lr=1e-3)
        optimizer_seg_model: Optimizer for segmentation model components (Adam, lr=1e-4)
        min_trace: Whether to minimize trace in loss function, switches to True after epoch 5
        alpha: Loss function weighting parameter, starts at 1.0, changes to 0.4 after epoch 5
        best_val_dice: Tracks best validation Dice score for model saving
        best_val_loss: Tracks best validation loss for model saving
        image_cm_patterns: List of patterns to identify confusion matrix layers in model
        device: Device to run training on (GPU if available, otherwise CPU)
    
    Methods:
        wandb_logging(): Sets up Weights & Biases monitoring if configured
        calculated_metrics(): Computes segmentation metrics (Dice, Jaccard, sensitivity, specificity)
        train_step(): Executes a single training step with gradient updates
        val_step(): Executes a single validation step
        training(): Runs the full training process for the specified number of epochs
        start(): Entry point to begin the training process
    """

    def __init__(self, model, train_dataset, valid_dataset, config_model, epochs=60, wandb_monitoring=None, all_ground_truths=False ):
        """
        Initialize the Crow_Seg_Training class with model and training parameters.
        
        Parameters:
        -----------
        model : tf.keras.Model
            The segmentation model with confusion matrix components to be trained.
        train_dataset : tf.data.Dataset
            Dataset for training containing (images, masks, annotator_ids) or 
            (images, masks, annotator_ids, original_masks).
        valid_dataset : tf.data.Dataset
            Dataset for validation with the same structure as train_dataset.
        config_model : dict
            Configuration dictionary with model parameters for WandB logging.
        epochs : int, default=60
            Number of training epochs.
        wandb_monitoring : None or list, default=None
            If not None, should be a list of [api_key, project_name, run_name] for Weights & Biases logging.
        all_ground_truths : bool, default=False
            If True, expects datasets to include original ground truth masks for metric calculation.
        """

        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.epochs = epochs
        self.optimizer_image_cm = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.optimizer_seg_model = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.min_trace = False
        self.alpha = 1
        self.best_val_dice = 0.0
        self.best_val_loss = float('inf')
        self.wandb_monitoring = wandb_monitoring
        self.all_ground_truths = all_ground_truths
        self.config_model = config_model
        self.image_cm_patterns = ['image_cm_', 'dense_annotator', 'feature_concat', 'dense_output', 'output_norm']
        self.run = None  # Initialize run here to handle WandB state
        self.device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0' # Detect GPU availability
        print(f"Using device: {self.device}")

    def wandb_logging(self):
        """
        Set up Weights & Biases logging if monitoring is enabled.
        
        Returns:
        --------
        str
            'Normal' if WandB monitoring is disabled, 'Tracking in WandB' if enabled.
            
        Raises:
        -------
        ValueError
            If wandb_monitoring is provided but not in the expected format.
        """
        
        if self.wandb_monitoring is None:
            return 'Normal'
        elif isinstance(self.wandb_monitoring, list) and len(self.wandb_monitoring) == 3 and all(isinstance(item, str) for item in self.wandb_monitoring):
            import wandb  # Ensure wandb is imported here
            wandb.login(key=self.wandb_monitoring[0])
            self.run = wandb.init(project=self.wandb_monitoring[1], name=self.wandb_monitoring[2], config=self.config_model)
            self.wandb_monitoring = True
            return 'Tracking in WandB'
        else:
            raise ValueError("The value is not valid. It must be None or a list with exactly three strings.")

    def calculated_metrics(self, y_true, y_pred):
        """
        Calculate segmentation evaluation metrics between true and predicted masks.
        
        Parameters:
        -----------
        y_true : tf.Tensor
            Ground truth segmentation masks.
        y_pred : tf.Tensor
            Predicted segmentation masks.
            
        Returns:
        --------
        tuple
            (dice, jaccard, sensitivity, specificity) metrics as TensorFlow tensors.
            NaN values are replaced with 0.0.
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

    def train_step(self):
        """
        Execute a single training step on the current batch.
        
        This method performs forward pass, loss calculation, and parameter updates
        with separate optimizers for confusion matrix and segmentation components.
        
        Returns:
        --------
        If self.orig_mask is not None:
            tuple: (loss, dice, jaccard, sensitivity, specificity)
        Else:
            float: The training loss value
        """
        self.masks = tf.argmax(self.masks, axis=1)

        with tf.GradientTape() as tape:
            y_pred, cms = self.model((self.images, self.ann_ids), training=True)

            loss = noisy_label_loss_tf(
                tf.transpose(y_pred, perm=[0, 3, 1, 2]),
                cms,
                self.masks,
                min_trace=self.min_trace,
                alpha=self.alpha
            )

        if not tf.math.is_nan(loss):
            # Calculate gradients for model parameters
            gradients = tape.gradient(loss, self.model.trainable_variables)
            
            # Separate parameters and gradients for Image_CM and ResUNet components
            image_cm_vars = []
            image_cm_grads = []
            seg_model_vars = []
            seg_model_grads = []
            
            for var, grad in zip(self.model.trainable_variables, gradients):
                if any(pattern in var.name for pattern in self.image_cm_patterns):
                    image_cm_vars.append(var)
                    image_cm_grads.append(grad)
                else:
                    seg_model_vars.append(var)
                    seg_model_grads.append(grad)
            
            # Update model weights using separate optimizers
            self.optimizer_image_cm.apply_gradients(zip(image_cm_grads, image_cm_vars))
            self.optimizer_seg_model.apply_gradients(zip(seg_model_grads, seg_model_vars))

        if self.orig_mask is not None:
            dice, jaccard, sensitivity, specificity = self.calculated_metrics(self.orig_mask, y_pred[..., 1:2])
            return loss, dice, jaccard, sensitivity, specificity
        else:
            return loss

    def val_step(self):
        """
        Execute a single validation step on the current batch.
        
        This method performs forward pass and loss calculation without parameter updates.
        
        Returns:
        --------
        If self.orig_mask is not None:
            tuple: (loss, dice, jaccard, sensitivity, specificity)
        Else:
            float: The validation loss value
        """
        self.masks = tf.argmax(self.masks, axis=1)

        y_pred, cms = self.model((self.images, self.ann_ids), training=False)

        loss = noisy_label_loss_tf(
            tf.transpose(y_pred, perm=[0, 3, 1, 2]),
            cms,
            self.masks,
            min_trace=self.min_trace,
            alpha=self.alpha
        )

        loss = tf.reduce_mean(loss) if not tf.math.is_nan(loss) else 0.0

        if self.orig_mask is not None:
            dice, jaccard, sensitivity, specificity = self.calculated_metrics(self.orig_mask, y_pred[..., 1:2])
            return loss, dice, jaccard, sensitivity, specificity
        else:
            return loss

    def training(self):
        """
        Run the full training loop for the specified number of epochs.
        
        This method handles:
        - Training and validation steps for each epoch
        - Metric calculation and logging
        - Model saving based on best validation performance
        - WandB logging if enabled
        - Performance parameter adjustments (alpha, min_trace) at epoch 5
        - Set Segmentation model learning rate to 1e-5 at epoch 45
        
        The best model is saved to './models/best_model.keras' based on either
        validation Dice score (if all_ground_truths=True) or validation loss.
        """
        
        type_training = self.wandb_logging()
        self.start_time = time.time()

        with tf.device(self.device):
        
            for epoch in range(self.epochs):
                print(f"\nEpoch {epoch + 1}/{self.epochs}")
    
                # Update parameters after epoch 5
                if epoch == 5:
                    self.min_trace = True
                    self.alpha = 0.4
                    print(f"Minimize trace activated!, Alpha updated to {self.alpha}")

                if epoch == 45:
                    self.optimizer_seg_model.learning_rate.assign(1e-5)
                    print(f"Learning rate updated to {1e-5}")
    
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
                        self.images, self.masks, self.ann_ids, self.orig_mask = data_batch
                    else:
                        self.images, self.masks, self.ann_ids = data_batch
                        self.orig_mask = None
    
                    if self.all_ground_truths:
                        batch_loss, batch_dice, batch_jaccard, batch_sensitivity, batch_specificity = self.train_step()
                        total_train_loss += batch_loss
                        total_train_dice += batch_dice
                        total_train_jaccard += batch_jaccard
                        total_train_sensitivity += batch_sensitivity
                        total_train_specificity += batch_specificity
                    else:
                        batch_loss = self.train_step()
                        total_train_loss += batch_loss
    
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
                        self.images, self.masks, self.ann_ids, self.orig_mask = data_batch
                    else:
                        self.images, self.masks, self.ann_ids = data_batch
                        self.orig_mask = None
    
                    if self.all_ground_truths:
                        batch_loss, batch_dice, batch_jaccard, batch_sensitivity, batch_specificity = self.val_step()
                        total_val_loss += batch_loss
                        total_val_dice += batch_dice
                        total_val_jaccard += batch_jaccard
                        total_val_sensitivity += batch_sensitivity
                        total_val_specificity += batch_specificity
                    else:
                        batch_loss = self.val_step()
                        total_val_loss += batch_loss
    
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
    
                tf.print(f"Training Loss: {'zero' if avg_train_loss == 0 else f'{avg_train_loss:.4f}'} | "
                      f"Validation Loss: {'zero' if avg_val_loss == 0 else f'{avg_val_loss:.4f}'} | "
                      f"Time: {elapsed_minutes}m {elapsed_seconds}s")
                
                if self.wandb_monitoring:
                    self.run.log({
                            'Training Loss': avg_train_loss,
                            'Validation Loss': avg_val_loss,
                        }, step=epoch)

                
                if self.all_ground_truths:
                    tf.print(f"Train_DICE: {'zero' if avg_train_dice == 0 else f'{avg_train_dice:.4f}'} | "
                            f"Valid_DICE: {'zero' if avg_val_dice == 0 else f'{avg_val_dice:.4f}'} | "
                            f"Train_Jaccard: {'zero' if avg_train_jaccard == 0 else f'{avg_train_jaccard:.4f}'} | "
                            f"Valid_Jaccard: {'zero' if avg_val_jaccard == 0 else f'{avg_val_jaccard:.4f}'} | "
                            f"Train_Sensitivity: {'zero' if avg_train_sensitivity == 0 else f'{avg_train_sensitivity:.4f}'} | "
                            f"Valid_Sensitivity: {'zero' if avg_val_sensitivity == 0 else f'{avg_val_sensitivity:.4f}'} | "
                            f"Train_Specificity: {'zero' if avg_train_specificity == 0 else f'{avg_train_specificity:.4f}'} | "
                            f"Valid_Specificity: {'zero' if avg_val_specificity == 0 else f'{avg_val_specificity:.4f}'} | "
                    )
                    if self.wandb_monitoring:
                        self.run.log({
                            'Training DICE': avg_train_dice,
                            'Validation DICE': avg_val_dice,
                            'Training Jaccard': avg_train_jaccard,
                            'Validation Jaccard': avg_val_jaccard,
                            'Training Sensitivity': avg_train_sensitivity,
                            'Validation Sensitivity': avg_val_sensitivity,
                            'Training Specificity': avg_train_specificity,
                            'Validation Specificity': avg_val_specificity
                        }, step=epoch)

        # Finish WandB run
        if self.wandb_monitoring:
            artifact = wandb.Artifact('best_model', type='model')
            artifact.add_file('./models/best_model.keras')
            self.run.log_artifact(artifact)
            self.run.finish()

        print("\n Training finished!")

    def start(self):
        """
        Start the training process.
        
        This is the main entry point for training after the class has been initialized
        Creates the models folder in the current directory in case it does not exist.
        """
        # Check if the directory exists, and if not, create it
        if not os.path.exists('./models'):
            os.makedirs('./models')

        self.training()