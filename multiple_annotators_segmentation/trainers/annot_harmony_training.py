import os
import time
import wandb
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from ..losses import TGCE_SS
from ..metrics import dice_metric, jaccard_metric, sensitivity_metric, specificity_metric



class Annot_Harmony_Training:
    """
    A class for training neural network models with multi-annotator datasets for image segmentation tasks.
    
    This class implements specialized training procedures for handling datasets with multiple annotator labels,
    using a consensus-based approach. It supports monitoring via Weights & Biases, custom loss functions,
    and various evaluation metrics.
    
    Attributes:
        model: The neural network model to be trained
        train_dataset: TensorFlow dataset for training
        valid_dataset: TensorFlow dataset for validation
        config_model: Dictionary containing model configuration parameters
        epochs: Number of training epochs
        wandb_monitoring: Configuration for Weights & Biases monitoring
        all_ground_truths: Boolean flag to determine if ground truth masks are available
        single_class: Index of a specific class for evaluation (if None, all classes are evaluated)
    """

    def __init__(self, model, train_dataset, valid_dataset, config_model, epochs=60, wandb_monitoring=None, all_ground_truths=False, single_class=None):
        """
        Initialize the training manager with the model and datasets.
        
        Args:
            model: The neural network model to be trained
            train_dataset: TensorFlow dataset for training
            valid_dataset: TensorFlow dataset for validation
            config_model: Dictionary containing model configuration parameters
            epochs: Number of training epochs (default: 60)
            wandb_monitoring: Configuration for Weights & Biases monitoring (default: None)
                             When not None, should be a list of [api_key, project_name, run_name]
            all_ground_truths: Boolean flag to determine if ground truth masks are available (default: False)
            single_class: Index of a specific class for evaluation (default: None)
        """
        
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.epochs = epochs
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.best_val_dice = 0.0
        self.best_val_loss = float('inf')
        self.wandb_monitoring = wandb_monitoring
        self.all_ground_truths = all_ground_truths
        self.single_class = single_class
        self.config_model = config_model
        self.run = None  # Initialize run here to handle WandB state
        self.device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0' # Detect GPU availability
        print(f"Using device: {self.device}")

    def wandb_logging(self):
        """
        Configure Weights & Biases logging if enabled.
        
        Sets up the W&B environment for experiment tracking based on the provided configuration.
        
        Returns:
            str: Status message indicating if W&B tracking is enabled
        
        Raises:
            ValueError: If wandb_monitoring is not None or a list with exactly three strings
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
        Calculate segmentation performance metrics for the model predictions.
        
        Computes Dice coefficient, Jaccard index (IoU), sensitivity, and specificity.
        Handles NaN values by replacing them with zeros.
        
        Args:
            y_true: Ground truth segmentation masks
            y_pred: Predicted segmentation masks
            
        Returns:
            tuple: (dice, jaccard, sensitivity, specificity) metrics
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
        
        Uses gradient tape to compute gradients and update model weights.
        Calculates metrics if ground truth is available.
        
        Returns:
            If all_ground_truths is True:
                tuple: (loss, dice, jaccard, sensitivity, specificity)
            Otherwise:
                float: loss value
        """
        
        with tf.GradientTape() as tape:
            y_pred = self.model(self.images, training=True)

            loss = self.loss_fn(self.masks, y_pred)

        if not tf.math.is_nan(loss):
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if self.orig_mask is not None:
            if isinstance(self.single_class, int):
                dice, jaccard, sensitivity, specificity = self.calculated_metrics(self.orig_mask, y_pred[..., self.single_class:self.single_class+1])
            else:
                dice, jaccard, sensitivity, specificity = self.calculated_metrics(self.orig_mask, y_pred)
            return loss, dice, jaccard, sensitivity, specificity
        else:
            return loss

    def val_step(self):
        """
        Execute a single validation step on the current batch.
        
        Calculates metrics if ground truth is available, without updating model weights.
        
        Returns:
            If all_ground_truths is True:
                tuple: (loss, dice, jaccard, sensitivity, specificity)
            Otherwise:
                float: loss value
        """
        

        y_pred = self.model(self.images, training=False)
        loss = self.loss_fn(self.masks, y_pred)

        loss = tf.reduce_mean(loss) if not tf.math.is_nan(loss) else 0.0

        if self.orig_mask is not None:
            if isinstance(self.single_class, int):
                dice, jaccard, sensitivity, specificity = self.calculated_metrics(self.orig_mask, y_pred[..., self.single_class:self.single_class+1])
            else:
                dice, jaccard, sensitivity, specificity = self.calculated_metrics(self.orig_mask, y_pred)
            return loss, dice, jaccard, sensitivity, specificity
        else:
            return loss

    def training(self):
        """
        Execute the complete training procedure.
        
        Runs the training and validation loops for the specified number of epochs.
        Implements learning rate scheduling, model checkpointing, and metric logging.
        Tracks and saves the best model based on validation metrics.
        """
        
        type_training = self.wandb_logging()
        self.start_time = time.time()

        # Check if the directory exists, and if not, create it
        if not os.path.exists('./models'):
            os.makedirs('./models')

        # Initialize TGCE_SS loss function with specific parameters
        self.loss_fn = TGCE_SS(q=0.48029, annotators=self.config_model['Num of annotators'], classes=self.config_model['Number of classes'], lambda_factor=0.0)

        with tf.device(self.device):
        
            for epoch in range(self.epochs):
                print(f"\nEpoch {epoch + 1}/{self.epochs}")
    
                # Update loss function parameters after epoch 5
                if epoch == 5:
                    self.loss_fn = TGCE_SS(q=0.48029, annotators=self.config_model['Num of annotators'], classes=self.config_model['Number of classes'], lambda_factor=1.0)
                    print(f"Loss's lambda factor updated to 1.0")

                # Reduce learning rate after epoch 45
                if epoch == 45:
                    self.optimizer.learning_rate.assign(1e-5)
                    print(f"Learning rate updated to {1e-5}")
                
                # Initialize tracking variables for training metrics
                total_train_loss = 0.0
                num_train_batches = 0
                total_train_dice = 0.0
                total_train_jaccard = 0.0
                total_train_sensitivity = 0.0
                total_train_specificity = 0.0
    
                # Training loop
                for data_batch in self.train_dataset:
                    data_length = len(data_batch)
                    if self.all_ground_truths and data_length == 3:
                        self.images, self.masks, self.orig_mask = data_batch
                    else:
                        self.images, self.masks = data_batch
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
    
                # Calculate average training metrics
                avg_train_loss = total_train_loss / num_train_batches
                avg_train_dice = total_train_dice / num_train_batches
                avg_train_jaccard = total_train_jaccard / num_train_batches
                avg_train_sensitivity = total_train_sensitivity / num_train_batches
                avg_train_specificity = total_train_specificity / num_train_batches
    
                # Initialize tracking variables for validation metrics
                total_val_loss = 0.0
                num_val_batches = 0
                total_val_dice = 0.0
                total_val_jaccard = 0.0
                total_val_sensitivity = 0.0
                total_val_specificity = 0.0
    
                # Validation loop
                for data_batch in self.valid_dataset:
                    data_length = len(data_batch)
                    if self.all_ground_truths and data_length == 3:
                        self.images, self.masks, self.orig_mask = data_batch
                    else:
                        self.images, self.masks = data_batch
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
    
                # Calculate average validation metrics
                avg_val_loss = total_val_loss / num_val_batches
                avg_val_dice = total_val_dice / num_val_batches
                avg_val_jaccard = total_val_jaccard / num_val_batches
                avg_val_sensitivity = total_val_sensitivity / num_val_batches
                avg_val_specificity = total_val_specificity / num_val_batches
    
                # Save best model based on different criteria
                if self.all_ground_truths:
                    # Save if Dice coefficient improved (higher is better)
                    if avg_val_dice > self.best_val_dice:
                        self.best_val_dice = avg_val_dice
                        self.model.save('./models/best_model.keras')
                else:
                    # Save if loss improved (lower is better)
                    if avg_val_loss < self.best_val_loss:
                        self.best_val_loss = avg_val_loss
                        self.model.save('./models/best_model.keras')
    
                # Calculate elapsed time
                elapsed_time = time.time() - self.start_time
                elapsed_minutes = int(elapsed_time // 60)
                elapsed_seconds = int(elapsed_time % 60)
    
                # Print training and validation loss
                tf.print(f"Training Loss: {'zero' if avg_train_loss == 0 else f'{avg_train_loss:.5f}'} | "
                      f"Validation Loss: {'zero' if avg_val_loss == 0 else f'{avg_val_loss:.5f}'} | "
                      f"Time: {elapsed_minutes}m {elapsed_seconds}s")
                
                # Log metrics to Weights & Biases if enabled
                if self.wandb_monitoring:
                    self.run.log({
                            'Training Loss': avg_train_loss,
                            'Validation Loss': avg_val_loss,
                        }, step=epoch)

                # Print additional metrics if ground truth is available
                if self.all_ground_truths:
                    tf.print(f"Train_DICE: {'zero' if avg_train_dice == 0 else f'{avg_train_dice:.5f}'} | "
                            f"Valid_DICE: {'zero' if avg_val_dice == 0 else f'{avg_val_dice:.5f}'} | "
                            f"Train_Jaccard: {'zero' if avg_train_jaccard == 0 else f'{avg_train_jaccard:.5f}'} | "
                            f"Valid_Jaccard: {'zero' if avg_val_jaccard == 0 else f'{avg_val_jaccard:.5f}'} | "
                            f"Train_Sensitivity: {'zero' if avg_train_sensitivity == 0 else f'{avg_train_sensitivity:.5f}'} | "
                            f"Valid_Sensitivity: {'zero' if avg_val_sensitivity == 0 else f'{avg_val_sensitivity:.5f}'} | "
                            f"Train_Specificity: {'zero' if avg_train_specificity == 0 else f'{avg_train_specificity:.5f}'} | "
                            f"Valid_Specificity: {'zero' if avg_val_specificity == 0 else f'{avg_val_specificity:.5f}'} | "
                    )
                    # Log additional metrics to Weights & Biases if enabled
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

        # Finish WandB run and log the best model as an artifact
        if self.wandb_monitoring:
            artifact = wandb.Artifact('best_model', type='model')
            artifact.add_file('./models/best_model.keras')
            self.run.log_artifact(artifact)
            self.run.finish()

        print("\n Training finished!")

    def start(self):
        """
        Start the training process.
        
        Creates the models directory if it doesn't exist and
        initiates the training loop.
        """
        # Check if the directory exists, and if not, create it
        if not os.path.exists('./models'):
            os.makedirs('./models')
        self.training()