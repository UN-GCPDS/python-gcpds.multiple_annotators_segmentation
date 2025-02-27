import tensorflow as tf
from tensorflow.keras.losses import Loss

class TGCE_SS(Loss):
    """
    Truncated Generalized Cross Entropy with Self-Supervised learning loss function.
    
    This loss function is designed for multi-annotator segmentation tasks, handling
    varying levels of annotator reliability. It extends the traditional cross-entropy
    by incorporating the Tukey's biweight function and a self-supervision mechanism
    that automatically learns to weight each annotator's contribution.
    
    Attributes:
        q (float): Parameter controlling the robustness of the loss. Lower values
                   make the loss more robust to noisy annotations. Default is 0.1.
        R (int): Number of annotators in the dataset. Default is 3.
        K_ (int): Number of segmentation classes. Default is 2.
        lambda_factor (float): Scaling factor for annotator reliability scores. 
                              Default is 1.0.
        smooth (float): Small value to avoid numerical instability. Default is 1e-5.
    """
    def __init__(self, q=0.1, name='TGCE_SS', annotators=3, classes=2, lambda_factor=1.0, smooth=1e-5, **kwargs):
        """
        Initialize the TGCE_SS loss function.
        
        Args:
            q (float, optional): Robustness parameter for Tukey's loss. 
                                 Default is 0.1.
            name (str, optional): Name of the loss function. Default is 'TGCE_SS'.
            annotators (int, optional): Number of annotators. Default is 3.
            classes (int, optional): Number of segmentation classes. Default is 2.
            lambda_factor (float, optional): Scaling factor for annotator weights. 
                                           Default is 1.0.
            smooth (float, optional): Smoothing factor to avoid division by zero. 
                                     Default is 1e-5.
            **kwargs: Additional arguments passed to the base Loss class.
        """
        self.q = q
        self.R = annotators
        self.K_ = classes
        self.smooth = smooth
        self.lambda_factor = lambda_factor
        super().__init__(name=name, **kwargs)
        
    def call(self, y_true, y_pred):
        """
        Calculate the TGCE_SS loss between ground truth and predictions.
        
        This method computes the loss by:
        1. Reshaping ground truth to separate annotator dimensions
        2. Extracting annotator reliability scores from the predictions
        3. Computing the main loss terms that balance between annotator reliability 
           and prediction accuracy
        
        Args:
            y_true (tf.Tensor): Ground truth segmentation masks from multiple annotators,
                                 with shape [batch, height, width, classes*annotators].
            y_pred (tf.Tensor): Model predictions, containing both segmentation 
                                outputs (first classes channels) and annotator 
                                reliability scores (remaining channels).
                                
        Returns:
            tf.Tensor: The computed TGCE_SS loss value (scalar).
        """
        # Obtain dynamic shape of y_true for proper reshaping
        shape_y_true = tf.shape(y_true)
        
        # Reshape y_true to separate class and annotator dimensions
        # From [batch, height, width, classes*annotators] to [batch, height, width, classes, annotators]
        y_true = tf.reshape(y_true, (shape_y_true[0], shape_y_true[1], shape_y_true[2], self.K_, self.R))
        
        # Extract annotator reliability scores from the predictions and apply scaling
        Lambda_r = y_pred[..., self.K_:] * self.lambda_factor  # Annotators reliability
        
        # Extract segmentation predictions (first K_ channels)
        y_pred_ = y_pred[..., :self.K_]  # Segmented images from model
        
        # Get batch size and spatial dimensions
        N, W, H = tf.shape(y_pred_)[0], tf.shape(y_pred_)[1], tf.shape(y_pred_)[2]
        
        # Expand predictions to match annotator dimension
        y_pred_ = tf.expand_dims(y_pred_, axis=-1)
        y_pred_ = tf.repeat(y_pred_, repeats=self.R, axis=-1)  # Repeat predictions for each annotator
        
        # Define small constant to avoid numerical issues
        epsilon = 1e-8  
        
        # Clip prediction values to avoid extreme logarithms
        y_pred_ = tf.clip_by_value(y_pred_, epsilon, 1.0 - epsilon)  
        
        # Calculate first term of the loss: weighted prediction error for each annotator
        term_r = tf.reduce_mean(
            tf.multiply(
                y_true, 
                (tf.ones((N, W, H, self.K_, self.R)) - tf.pow(y_pred_, self.q)) / (self.q + epsilon + self.smooth)
            ), 
            axis=-2  # Average across class dimension
        )
        
        # Calculate second term: penalty for low annotator reliability
        term_c = tf.multiply(
            tf.ones((N, W, H, self.R)) - Lambda_r,  # Complement of reliability
            (tf.ones((N, W, H, self.R)) - tf.pow(
                (1 / self.K_ + self.smooth) * tf.ones((N, W, H, self.R)), 
                self.q
            )) / (self.q + epsilon + self.smooth)
        )
        
        # Combine terms with annotator reliability weights
        TGCE_SS = tf.reduce_mean(tf.multiply(Lambda_r, term_r) + term_c)
        
        # Handle potential NaN values in the loss
        if tf.reduce_any(tf.math.is_nan(TGCE_SS)):
            TGCE_SS = tf.where(tf.math.is_nan(TGCE_SS), tf.constant(1e-8), TGCE_SS)  # Replace NaN with 1e-8
            print("\nInitializing TGCE_SS \n")
            
        return TGCE_SS
        
    def get_config(self):
        """
        Return the configuration of the loss function.
        
        This method is used for serialization when saving and loading models.
        
        Returns:
            dict: Dictionary containing the configuration of the loss function.
        """
        base_config = super().get_config()
        return {**base_config, "q": self.q, "R": self.R, "K_": self.K_, 
                "lambda_factor": self.lambda_factor, "smooth": self.smooth}