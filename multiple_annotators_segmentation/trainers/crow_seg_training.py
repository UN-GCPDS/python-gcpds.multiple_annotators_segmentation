import os
import time
import wandb
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from ..losses import noisy_label_loss_tf
from ..metrics import dice_metric, jaccard_metric, sensitivity_metric, specificity_metric



class Crow_Seg_Training:
    """
    A training manager for segmentation models with confusion matrix learning.
    
    This class handles the training process for segmentation models that incorporate 
    a crowdsourcing-based approach using confusion matrices to model annotator accuracy.
    It supports training with noisy labels from multiple annotators and provides robust
    evaluation metrics when ground truth segmentation masks are available.
    
    The class manages separate optimizers for the image confusion matrix components and 
    the segmentation model components, applies dynamic parameter adjustments during training,
    and supports optional Weights & Biases monitoring.
    
    Parameters
    ----------
    model : tf.keras.Model
        The segmentation model with confusion matrix capability.
    train_dataset : tf.data.Dataset
        Training dataset yielding (images, masks, annotator_ids) or 
        (images, masks, annotator_ids, ground_truth_masks) tuples.
    valid_dataset : tf.data.Dataset
        Validation dataset with the same structure as train_dataset.
    config_model : dict
        Configuration parameters for the model to be logged in WandB.
    epochs : int, optional
        Number of training epochs (default: 60).
    wandb_monitoring : list or None, optional
        If provided, a list with [api_key, project_name, run_name] for WandB tracking.
    all_ground_truths : bool, optional
        Whether the datasets include ground truth masks for evaluation (default: False).
    single_class : int or None, optional
        If specified, evaluate metrics only on this specific class (default: None).
    
    Attributes
    ----------
    best_val_dice : float
        Best validation Dice coefficient achieved during training.
    best_val_loss : float
        Best validation loss achieved during training.
    alpha : float
        Weight parameter for the loss function, adjusted during training.
    min_trace : bool
        Flag to enable trace minimization in the loss function, activated after epoch 5.
    device : str
        Device used for training ('/GPU:0' or '/CPU:0').
    
    Methods
    -------
    start()
        Main entry point to begin the training process.
    training()
        Run the full training loop with train and validation steps.
    train_step()
        Execute a single training step with gradient updates.
    val_step()
        Execute a single validation step without gradient updates.
    calculated_metrics(y_true, y_pred)
        Calculate segmentation quality metrics between true and predicted masks.
    wandb_logging()
        Set up Weights & Biases logging if monitoring is enabled.
    """

    def __init__(self, model, train_dataset, valid_dataset, config_model, epochs=60, wandb_monitoring=None, all_ground_truths=False, single_class=None):
        """
        Initialize the Crow_Seg_Training instance with model and training parameters.
        
        Parameters
        ----------
        model : tf.keras.Model
            The segmentation model with confusion matrix capability.
        train_dataset : tf.data.Dataset
            Training dataset yielding (images, masks, annotator_ids) tuples or
            (images, masks, annotator_ids, ground_truth_masks) tuples when all_ground_truths=True.
        valid_dataset : tf.data.Dataset
            Validation dataset with the same structure as train_dataset.
        config_model : dict
            Configuration parameters for the model to be logged in WandB.
        epochs : int, optional
            Number of training epochs. Default is 60.
        wandb_monitoring : list or None, optional
            If provided, must be a list containing [api_key, project_name, run_name] for 
            Weights & Biases tracking. Default is None (no monitoring).
        all_ground_truths : bool, optional
            Whether datasets include ground truth masks for evaluation metrics calculation.
            When True, datasets should yield 4-tuples including original masks.
            Default is False.
        single_class : int or None, optional
            If specified, calculate metrics only for this specific class index.
            Default is None (evaluate on all classes).
        
        Attributes
        ----------
        optimizer_image_cm : tf.keras.optimizers.Adam
            Optimizer for the confusion matrix components with learning rate 1e-3.
        optimizer_seg_model : tf.keras.optimizers.Adam
            Optimizer for the segmentation model components with learning rate 1e-4.
        min_trace : bool
            Flag for loss function trace minimization, initially False.
        alpha : float
            Weight parameter for the loss function, initially 1.
        best_val_dice : float
            Best validation Dice coefficient, initially 0.0.
        best_val_loss : float
            Best validation loss, initially infinity.
        image_cm_patterns : list
            List of string patterns to identify confusion matrix component variables.
        run : wandb.Run or None
            WandB run instance if monitoring is enabled.
        device : str
            Detected device for training ('/GPU:0' or '/CPU:0').
        """

        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.epochs = epochs
        self.optimizer_image_cm = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.optimizer_seg_model = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.min_trace = False
        self.single_class = single_class
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
        
        This method performs the following operations:
        1. Converts masks to class indices using argmax
        2. Executes forward pass through the model with training=True
        3. Calculates the noisy label loss using confusion matrices
        4. Computes gradients for both model components
        5. Applies separate optimizers to confusion matrix and segmentation components
        6. Calculates evaluation metrics if ground truth masks are available
        
        The method dynamically handles variables using two separate optimizers:
        - optimizer_image_cm: Updates confusion matrix component variables
        - optimizer_seg_model: Updates segmentation model variables
        
        Variables are categorized based on name patterns in self.image_cm_patterns.
        
        Returns
        -------
        tuple or float
            If self.orig_mask is available (all_ground_truths=True):
                Returns (loss, dice, jaccard, sensitivity, specificity)
            Otherwise:
                Returns only the training loss value
            
        Notes
        -----
        This method expects the following instance variables to be set:
        - self.images: Input images batch
        - self.masks: Annotator masks batch
        - self.ann_ids: Annotator IDs batch
        - self.orig_mask: Ground truth masks batch (optional)
        
        NaN loss values will not trigger gradient updates to prevent model instability.
        When single_class is specified, metrics are calculated only for that class.
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
            if isinstance(self.single_class,int):
                dice, jaccard, sensitivity, specificity = self.calculated_metrics(self.orig_mask, y_pred[..., self.single_class:self.single_class+1])
            else:
                dice, jaccard, sensitivity, specificity = self.calculated_metrics(self.orig_mask, y_pred)
            return loss, dice, jaccard, sensitivity, specificity
        else:
            return loss

    def val_step(self):
        """
        Execute a single validation step on the current batch.
        
        This method performs the following operations:
        1. Converts masks to class indices using argmax
        2. Executes forward pass through the model with training=False
        3. Calculates the noisy label loss using confusion matrices
        4. Calculates evaluation metrics if ground truth masks are available
        
        Unlike train_step, no gradient computation or parameter updates occur
        during validation, making this method more efficient for evaluation.
        
        Returns
        -------
        tuple or float
            If self.orig_mask is available (all_ground_truths=True):
                Returns (loss, dice, jaccard, sensitivity, specificity)
            Otherwise:
                Returns only the validation loss value
        
        Notes
        -----
        This method expects the following instance variables to be set:
        - self.images: Input images batch
        - self.masks: Annotator masks batch
        - self.ann_ids: Annotator IDs batch
        - self.orig_mask: Ground truth masks batch (optional)
        
        NaN loss values are replaced with 0.0 to ensure stable reporting.
        When single_class is specified, metrics are calculated only for that class.
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
            if isinstance(self.single_class,int):
                dice, jaccard, sensitivity, specificity = self.calculated_metrics(self.orig_mask, y_pred[..., self.single_class:self.single_class+1])
            else:
                dice, jaccard, sensitivity, specificity = self.calculated_metrics(self.orig_mask, y_pred)
            return loss, dice, jaccard, sensitivity, specificity
        else:
            return loss

    def training(self):
        """
        Run the full training loop for the specified number of epochs.
        
        This method conducts a complete training process including:
        1. Setup of WandB monitoring if enabled
        2. Execution of training and validation steps for each epoch
        3. Calculation and logging of performance metrics
        4. Dynamic adjustment of training parameters at specific epochs:
        - Epoch 5: Enables min_trace and reduces alpha to 0.4
        - Epoch 45: Reduces segmentation model learning rate to 1e-5
        5. Saving the best model based on validation performance
        6. Uploading model artifacts to WandB if monitoring is enabled
        
        The best model selection criterion depends on the all_ground_truths setting:
        - If True: Model with highest validation Dice score is saved
        - If False: Model with lowest validation loss is saved
        
        Returns
        -------
        None
            The method doesn't return any value but saves the best model to './models/best_model.keras'
        
        Notes
        -----
        The method tracks multiple metrics during training when ground truth masks are available:
        - Dice coefficient
        - Jaccard index (IoU)
        - Sensitivity (Recall)
        - Specificity
        
        Time elapsed is calculated and displayed for each epoch to monitor training progress.
        All metrics are logged to WandB if monitoring is enabled, providing comprehensive 
        visualization of the training process.
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