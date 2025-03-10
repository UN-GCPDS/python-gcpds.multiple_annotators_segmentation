import tensorflow as tf
from tensorflow import keras
from functools import partial
import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model
from keras.layers import Layer, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects

# ResUNet Model Backbone
from classification_models.keras import Classifiers  # Requires: pip install git+https://github.com/qubvel/classification_models.git

ResNet34, preprocess_input = Classifiers.get('resnet34')

# Definition of layers for the neural network structure

DefaultConv2D = partial(layers.Conv2D,
                        kernel_size=3, activation='relu', padding="same")

DefaultPooling = partial(layers.MaxPool2D,
                        pool_size=2)

DilatedConv = partial(layers.Conv2D,
                        kernel_size=3, activation='relu', padding="same", dilation_rate=10, name="DilatedConv")

upsample = partial(layers.UpSampling2D, (2,2))


def kernel_initializer(seed):
    """Returns a Glorot uniform initializer for kernel weights.

    Glorot uniform initializer, also known as Xavier uniform initializer,
    is commonly used to initialize the weights of kernels in neural network layers.
    It draws samples from a uniform distribution within a certain range,
    calculated to keep the variance of the weights constant across layers.
    This initializer is useful for training deep neural networks.

    Parameters:
        seed (int): Random seed for reproducibility.

    Returns:
        A Glorot uniform initializer for kernel weights.

    """
    return tf.keras.initializers.GlorotUniform(seed=seed)

class SparseSoftmax(Layer):
    """Custom layer implementing the sparse softmax activation function.

    This layer computes the softmax activation function for a given input tensor,
    handling sparse input efficiently.

    Methods
    ----------
    build(input_shape)
    call(x)
    compute_output_shape(input_shape)

    """

    def __init__(self, name='SparseSoftmax', **kwargs):
        """Initializes the SparseSoftmax layer.

        Parameters:
            **kwargs: Additional arguments to be passed to the parent class.

        """
        super(SparseSoftmax, self).__init__(name=name,**kwargs)

    def build(self, input_shape):
        """Builds the layer.

        Parameters:
            input_shape (tuple): Shape of the input tensor.

        """
        super(SparseSoftmax, self).build(input_shape)

    def call(self, x):
        """Computes the output of the layer.

        Parameters:
            x (tensor): Input tensor.

        Returns:
            A tensor representing the output of the softmax activation function.

        """
        e_x = K.exp(x - K.max(x, axis=-1, keepdims=True))
        sum_e_x = K.sum(e_x, axis=-1, keepdims=True)
        output = e_x / (sum_e_x + K.epsilon())
        return output

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        Parameters:
            input_shape (tuple): Shape of the input tensor.

        Returns:
            The same shape as the input tensor.

        """
        return input_shape

tf.keras.utils.get_custom_objects()['sparse_softmax'] = SparseSoftmax()

def residual_block(x, filters, kernel_initializer, block_name):
    """
    Defines a residual block for a convolutional neural network.

    This function implements a standard residual block, which consists of two 
    convolutional layers with batch normalization and a skip connection that 
    adds the input to the output of the block. The block is designed to ease 
    the training of deep networks by allowing gradients to flow more easily 
    through the network.

    Args:
        x (tf.Tensor): Input tensor from the previous layer.
        filters (int): Number of filters for the convolutional layers.
        kernel_initializer: Initializer for the convolutional kernels.
        block_name (str): Name for the block, used to name the layers within the block.

    Returns:
        tf.Tensor: Output tensor of the residual block.

    Example:
        >>> from tensorflow.keras.layers import Input
        >>> input_tensor = Input(shape=(256, 256, 3))
        >>> output_tensor = residual_block(input_tensor, filters=64, kernel_initializer='glorot_uniform', block_name='block1')
        >>> print(output_tensor.shape)
        (None, 256, 256, 64)

    Note:
        - The `kernel_initializer` should be a valid Keras initializer.
        - The `block_name` should be single for each block to avoid naming conflicts.

    See Also:
        ResUNet: A UNet backbone implementation that can use this residual block.
        DefaultConv2D: A custom convolutional layer function used in this block.
    """
    shortcut = x
    x = DefaultConv2D(filters, kernel_initializer=kernel_initializer, name=f'Conv_{block_name}_1')(x)
    x = layers.BatchNormalization(name=f'Batch_{block_name}_1')(x)
    x = DefaultConv2D(filters, kernel_initializer=kernel_initializer, name=f'Conv_{block_name}_2')(x)
    x = layers.BatchNormalization(name=f'Batch_{block_name}_2')(x)
    x = layers.Add(name=f'ResAdd_{block_name}')([shortcut, x])
    x = layers.Activation('relu', name=f'ResAct_{block_name}')(x)
    return x

def UNet(inputs, out_channels, n_annotators, activation):
    """
    Implements a U-Net architecture for segmentation tasks with multiple annotators support.
    
    This is a standard U-Net architecture with encoder and decoder paths, skip connections,
    and an additional output branch for annotator scoring. The network follows a symmetric
    structure with downsampling and upsampling operations.
    
    Args:
        inputs (tf.Tensor): Input tensor representing the image to segment.
        out_channels (int): Number of output channels (classes) for segmentation.
        n_scorers (int): Number of annotators in the dataset.
        activation (str): Activation function for the segmentation output 
                          (e.g., 'softmax', 'sigmoid').
    
    Returns:
        tf.Tensor: Concatenated output tensor containing both the segmentation maps
                   and the annotator scoring weights.
    
    Note:
        The architecture includes a main branch for segmentation and a separate branch
        for annotator scoring using sparse softmax activation.
    """
    # Encoder - Downsampling path with increasing filter numbers
    # Each level follows: Conv -> Dropout -> BatchNorm -> Conv -> Dropout -> BatchNorm -> Pooling

    x = layers.BatchNormalization(name='Batch00')(inputs)
    x = DefaultConv2D(8, kernel_initializer=kernel_initializer(34), name='Conv10')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization(name='Batch10')(x)
    x = level_1 = DefaultConv2D(8, kernel_initializer=kernel_initializer(4), name='Conv11')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization(name='Batch11')(x)
    x = DefaultPooling(name='Pool10')(x)  # 128x128 -> 64x64

    x = DefaultConv2D(16, kernel_initializer=kernel_initializer(56), name='Conv20')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization(name='Batch20')(x)
    x = level_2 = DefaultConv2D(16, kernel_initializer=kernel_initializer(32), name='Conv21')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization(name='Batch22')(x)
    x = DefaultPooling(name='Pool20')(x)  # 64x64 -> 32x32

    x = DefaultConv2D(32, kernel_initializer=kernel_initializer(87), name='Conv30')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization(name='Batch30')(x)
    x = level_3 = DefaultConv2D(32, kernel_initializer=kernel_initializer(30), name='Conv31')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization(name='Batch31')(x)
    x = DefaultPooling(name='Pool30')(x)  # 32x32 -> 16x16

    x = DefaultConv2D(64, kernel_initializer=kernel_initializer(79), name='Conv40')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization(name='Batch40')(x)
    x = level_4 = DefaultConv2D(64, kernel_initializer=kernel_initializer(81), name='Conv41')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization(name='Batch41')(x)
    x = DefaultPooling(name='Pool40')(x)  # 16x16 -> 8x8

    # Decoder - Upsampling path with decreasing filter numbers
    # Each level follows: Conv -> Dropout -> BatchNorm -> Conv -> Dropout -> BatchNorm
    x = DefaultConv2D(128, kernel_initializer=kernel_initializer(89), name='Conv50')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization(name='Batch50')(x)
    x = DefaultConv2D(128, kernel_initializer=kernel_initializer(42), name='Conv51')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization(name='Batch51')(x)

    # Upsampling and skip connection with level_4 features
    x = upsample(name='Up60')(x)  # 8x8 -> 16x16
    x = layers.Concatenate(name='Concat60')([level_4, x])
    x = DefaultConv2D(64, kernel_initializer=kernel_initializer(91), name='Conv60')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization(name='Batch60')(x)
    x = DefaultConv2D(64, kernel_initializer=kernel_initializer(47), name='Conv61')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization(name='Batch61')(x)

    # Upsampling and skip connection with level_3 features
    x = upsample(name='Up70')(x)  # 16x16 -> 32x32
    x = layers.Concatenate(name='Concat70')([level_3, x])
    x = DefaultConv2D(32, kernel_initializer=kernel_initializer(21), name='Conv70')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization(name='Batch70')(x)
    x = DefaultConv2D(32, kernel_initializer=kernel_initializer(96), name='Conv71')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization(name='Batch71')(x)

    # Upsampling and skip connection with level_2 features
    x = upsample(name='Up80')(x)  # 32x32 -> 64x64
    x = layers.Concatenate(name='Concat80')([level_2, x])
    x = DefaultConv2D(16, kernel_initializer=kernel_initializer(96), name='Conv80')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization(name='Batch80')(x)
    x = DefaultConv2D(16, kernel_initializer=kernel_initializer(98), name='Conv81')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization(name='Batch81')(x)

    # Upsampling and skip connection with level_1 features
    x = upsample(name='Up90')(x)  # 64x64 -> 128x128
    x = layers.Concatenate(name='Concat90')([level_1, x])
    x = DefaultConv2D(8, kernel_initializer=kernel_initializer(35), name='Conv90')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization(name='Batch90')(x)
    x = DefaultConv2D(8, kernel_initializer=kernel_initializer(7), name='Conv91')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization(name='Batch91')(x)

    # Final segmentation output branch with specified activation
    xy = DefaultConv2D(out_channels, kernel_size=(1, 1), activation=activation,
                      kernel_initializer=kernel_initializer(42), name='Conv200')(x)
                      
    # Annotator scoring branch with sparse softmax activation
    x_lambda = DilatedConv(n_annotators, kernel_size=(1, 1), activation='sparse_softmax',
                        kernel_initializer=kernel_initializer(42), name='DilatedConv200-Lambda')(x)

    # Combine segmentation outputs and annotator scoring
    y = layers.Concatenate(name='Concat200')([xy, x_lambda])

    return y
    

def ResUNet(inputs, out_channels, n_annotators, activation):
    """
    Implements a ResNet-based U-Net architecture for segmentation with multiple annotators support.
    
    This network uses a pre-trained ResNet34 as the encoder backbone and adds a decoder path
    with skip connections. It includes residual blocks in the decoder path and a dual-branch
    output for both segmentation and annotator scoring.
    
    Args:
        inputs (tf.Tensor): Input tensor representing the image to segment.
        out_channels (int): Number of output channels (classes) for segmentation.
        n_annotators (int): Number of annotators in the dataset.
        activation (str): Activation function for the segmentation output 
                          (e.g., 'softmax', 'sigmoid').
    
    Returns:
        tf.Tensor: Concatenated output tensor containing both the segmentation maps
                   and the annotator scoring weights.
    
    Note:
        The ResNet34 backbone is pre-trained on ImageNet and its BatchNormalization 
        layers remain trainable while other layers are frozen.
    """
    # Backbone (ResNet34) - Use pre-trained weights from ImageNet
    backbone = ResNet34(
        input_tensor=inputs,
        include_top=False,
        weights='imagenet')

    # Freeze backbone layers except BatchNormalization layers
    # This allows the model to adapt to the new domain while retaining learned features
    for layer in backbone.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    # Extract feature maps from different levels of the ResNet backbone
    level_1 = backbone.layers[5].output   # 128x128x64
    level_2 = backbone.layers[37].output  # 64x64x64
    level_3 = backbone.layers[74].output  # 32x32x128
    level_4 = backbone.layers[129].output # 16x16x256
    x = backbone.layers[157].output       # 8x8x512

    # Decoder - Upsampling path with skip connections to encoder features
    # First upsampling block
    x = upsample(name='Up60')(x)
    x = layers.Concatenate(name='Concat60')([level_4, x])
    x = DefaultConv2D(256, kernel_initializer=kernel_initializer(91) , name='Conv60')(x)
    x = residual_block(x, 256, kernel_initializer(47), '60')  # Apply residual block

    # Second upsampling block
    x = upsample(name='Up70')(x)
    x = layers.Concatenate(name='Concat70')([level_3, x])
    x = DefaultConv2D(128, kernel_initializer=kernel_initializer(21), name='Conv70')(x)
    x = residual_block(x, 128, kernel_initializer(96), '70')  # Apply residual block

    # Third upsampling block
    x = upsample(name='Up80')(x)
    x = layers.Concatenate(name='Concat80')([level_2, x])
    x = DefaultConv2D(64, kernel_initializer=kernel_initializer(96), name='Conv80')(x)
    x = residual_block(x, 64, kernel_initializer(98), '80')  # Apply residual block

    # Fourth upsampling block
    x = upsample(name='Up90')(x)
    x = layers.Concatenate(name='Concat90')([level_1, x])
    x = DefaultConv2D(32, kernel_initializer=kernel_initializer(35), name='Conv90')(x)
    x = residual_block(x, 32, kernel_initializer(7), '90')  # Apply residual block

    # Final upsampling block - connect back to input
    x = upsample(name='Up100')(x)
    x = layers.Concatenate(name='Concat100')([inputs, x])
    x = DefaultConv2D(16, kernel_initializer=kernel_initializer(45), name='Conv100')(x)
    x = residual_block(x, 16, kernel_initializer(7), '100')  # Apply residual block

    # Final segmentation output branch with specified activation
    xy = DefaultConv2D(out_channels, kernel_size=(1, 1), activation=activation,
                      kernel_initializer=kernel_initializer(42), name='Conv200')(x)
    
    # Annotator scoring branch with sparse softmax activation
    x_lambda = DilatedConv(n_annotators, kernel_size=(1, 1), activation='sparse_softmax',
                        kernel_initializer=kernel_initializer(42), name='DilatedConv200-Lambda')(x)

    # Combine segmentation outputs and annotator scoring
    y = layers.Concatenate(name='Concat200')([xy, x_lambda])

    return y

def Annot_Harmony_Model(num_annotators=3, class_no=2, input_shape=(512, 512, 3), seg_model='ResUNet', activation='softmax'):
    """
    Creates a segmentation model that can harmonize annotations from multiple annotators.
    
    This function builds a model that can handle multi-annotator segmentation datasets by
    learning to weight the importance of each annotator's input. It supports different
    segmentation backbone architectures (UNet or ResUNet).
    
    Args:
        num_annotators (int, optional): Number of annotators in the dataset. Defaults to 3.
        class_no (int, optional): Number of segmentation classes. Defaults to 2.
        input_shape (tuple, optional): Input image dimensions (height, width, channels). 
                                       Defaults to (512, 512, 3).
        seg_model (str, optional): Segmentation model architecture to use ('UNet' or 'ResUNet'). 
                                   Defaults to 'ResUNet'.
        activation (str, optional): Activation function for segmentation outputs. 
                                    Defaults to 'softmax'.
    
    Returns:
        tf.keras.Model: Compiled Annot_Harmony model.
        
    Raises:
        ValueError: If an unsupported segmentation model type is specified.
        
    Note:
        The model produces a combined output with both class segmentation maps and
        annotator scoring weights, allowing for annotation harmonization.
    """
    
    # Input layer definition
    image_input = layers.Input(shape=input_shape, name='image_input')

    # Select and build the appropriate segmentation backbone
    if seg_model == 'ResUNet':
        # Use ResNet-based UNet architecture
        seg_rm = ResUNet(image_input, out_channels=class_no, n_annotators=num_annotators, activation=activation)
    elif seg_model == 'UNet':
        # Use standard UNet architecture
        seg_rm = UNet(image_input, out_channels=class_no, n_annotators=num_annotators, activation=activation)
    else:
        # Raise error for unsupported model types
        raise ValueError(f"Unsupported segmentation model type: {seg_model}. Supported models are 'ResUNet' and 'UNet'.")

    # Create and return the final model
    return Model(image_input, seg_rm, name='Annot_Harmony')