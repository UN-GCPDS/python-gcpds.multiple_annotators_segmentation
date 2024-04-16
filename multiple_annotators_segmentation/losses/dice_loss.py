import tensorflow as tf 
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

class DiceCoefficient(Loss):
    
    """
    DiceCoefficient loss function.

    This loss function computes the Dice coefficient, a similarity metric commonly
    used in image segmentation tasks. It measures the overlap between the predicted
    segmentation mask (y_pred) and the ground truth mask (y_true).
    
    Methods
    ----------
    call(y_true, y_pred)
    get_config()
    """
    
    def __init__(self, smooth=1., target_class= None, name='DiceCoefficient', **kwargs):
        """
        Initializes the DiceCoefficient loss object.

        Parameters:
            smooth (float): A smoothing parameter to avoid division by zero. Defaults to 1.0.
            target_class (int or None): If specified, computes the Dice coefficient only for 
                the specified class index. If None, computes the average Dice coefficient 
                across all classes. Defaults to None.
            name (str): Name of the loss function. Defaults to 'DiceCoefficient'.
            **kwargs: Additional arguments passed to the parent class.
        """
        self.smooth = smooth
        self.target_class = target_class
        super().__init__(name=name,**kwargs)

    def call(self, y_true, y_pred):
        """
        Computes the Dice coefficient loss.

        Parameters:
            y_true (tensor): Ground truth segmentation masks.
            y_pred (tensor): Predicted segmentation masks.

        Returns:
            A tensor representing the Dice coefficient loss.
        """
        intersection = K.sum(y_true * y_pred, axis=[1,2])
        union = K.sum(y_true,axis=[1,2]) + K.sum(y_pred,axis=[1,2])
        dice_coef = -(2. * intersection + self.smooth) /(union + self.smooth)

        if self.target_class != None:
            dice_coef = tf.gather(dice_coef,
                                  self.target_class, axis=1)
        else:
            dice_coef = K.mean(dice_coef,axis=-1)

        return dice_coef

    def get_config(self,):
        """
        Gets the configuration of the loss function.

        Returns:
            A dictionary containing the configuration parameters of the loss function.
        """
        base_config = super().get_config()
        return {**base_config, "smooth": self.smooth,
                "target_class":self.target_class}