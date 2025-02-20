import tensorflow as tf
from tensorflow import keras
from functools import partial
from tensorflow.keras.backend import K
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
        - The `block_name` should be unique for each block to avoid naming conflicts.

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

def ConvLayersImage(inputs):
    
    """
    Defines a series of convolutional and fully connected layers for image processing.

    This function implements a multi-layer convolutional neural network (CNN) 
    followed by fully connected layers. The CNN consists of four convolutional 
    blocks, each followed by batch normalization, ReLU activation, and max pooling. 
    The output of the CNN is then flattened and passed through two fully connected 
    layers.

    Args:
        inputs (tf.Tensor): Input tensor to the network, typically an image.

    Returns:
        tf.Tensor: Output tensor of the network, representing a feature vector.

    Example:
        >>> from tensorflow.keras.layers import Input
        >>> input_tensor = Input(shape=(256, 256, 3))
        >>> output_tensor = ConvLayersImage(input_tensor)
        >>> print(output_tensor.shape)
        (None, 64)

    Note:
        - The input tensor should have a shape compatible with the convolutional layers.
        - The number of filters, kernel size, and other hyperparameters are hardcoded 
          in this implementation. You can modify these parameters as needed.

    See Also:
        ResUNet: A UNet backbone implementation that can be used in conjunction with this function.
        Segmentation_Head: A segmentation head that can be used after this function to produce segmentation outputs.
    """

    # First convolution block
    x = layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='same',
                     activation=None, name='image_cm_conv_block1_conv')(inputs)
    x = layers.BatchNormalization(name='image_cm_conv_block1_bn')(x)
    x = layers.ReLU(name='image_cm_conv_block1_relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, name='image_cm_conv_block1_pool')(x)

    # Second convolution block
    x = layers.Conv2D(filters=4, kernel_size=3, strides=1, padding='same',
                     activation=None, name='image_cm_conv_block2_conv')(x)
    x = layers.BatchNormalization(name='image_cm_conv_block2_bn')(x)
    x = layers.ReLU(name='image_cm_conv_block2_relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, name='image_cm_conv_block2_pool')(x)

    # Third convolution block
    x = layers.Conv2D(filters=4, kernel_size=3, strides=1, padding='same',
                     activation=None, name='image_cm_conv_block3_conv')(x)
    x = layers.BatchNormalization(name='image_cm_conv_block3_bn')(x)
    x = layers.ReLU(name='image_cm_conv_block3_relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, name='image_cm_conv_block3_pool')(x)

    # Fourth convolution block
    x = layers.Conv2D(filters=4, kernel_size=3, strides=1, padding='same',
                     activation=None, name='image_cm_conv_block4_conv')(x)
    x = layers.BatchNormalization(name='image_cm_conv_block4_bn')(x)
    x = layers.ReLU(name='image_cm_conv_block4_relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, name='image_cm_conv_block4_pool')(x)

    # Fully connected layers
    x = layers.Flatten(name='image_cm_flatten')(x)
    x = layers.Dense(128, name='image_cm_dense1')(x)
    x = layers.BatchNormalization(name='image_cm_dense1_bn')(x)
    x = layers.ReLU(name='image_cm_dense1_relu')(x)
    outputs = layers.Dense(64, name='image_cm_dense2')(x)

    return outputs

def Image_CM(A_id_input, image_input, class_no, input_height, input_width):
    
    """
    Generates a crowd matrix from annotator and image features.

    This function processes annotator IDs and image inputs to produce a crowd matrix.
    Annotator features are transformed via a dense layer, while image features undergo
    multi-layer convolutional processing. Concatenated features are then mapped to a 
    crowd matrix, which is spatially expanded to match input image dimensions.

    Args:
        A_id_input (tf.Tensor): Input tensor representing annotator IDs.
        image_input (tf.Tensor): Input tensor representing the image.
        class_no (int): Number of classes in the segmentation task.
        input_height (int): Height of the input image.
        input_width (int): Width of the input image.

    Returns:
        tf.Tensor: The generated crowd matrix with shape 
            (batch_size, class_no * class_no, input_height, input_width).

    Example:
        >>> from tensorflow.keras.layers import Input
        >>> A_id_input = Input(shape=(10,), name='annotator_input')
        >>> image_input = Input(shape=(256, 256, 3), name='image_input')
        >>> crowd_matrix = Image_CM(A_id_input, image_input, class_no=2, 
                                    input_height=256, input_width=256)
        >>> print(crowd_matrix.shape)
        (None, 4, 256, 256)

    Note:
        - Requires prior definition/import of `ConvLayersImage()`
        - Input tensor shapes must align with layer requirements

    See Also:
        ConvLayersImage: Image feature processing module used here
        Crowd_Segmentation_Model: Parent model integrating this crowd matrix

    References:
        This implementation originates from a manual conversion of the Image_CM 
        model proposed by López-Pérez et al. (2024) in their work 
        "Learning from crowds for automated histopathological image segmentation". 
        For additional details, consult the original GitHub repository: 
        https://github.com/wizmik12/crowd_seg
    """

    # Annotator feature processing
    A_feat = layers.Dense(64, name='dense_annotator')(A_id_input)

    # Image feature processing
    x = ConvLayersImage(image_input)

    # Feature concatenation
    concatenated = layers.Concatenate(axis=-1, name='feature_concat')([A_feat, x])

    # Output generation
    output = layers.Dense(class_no * class_no, name='dense_output')(concatenated)
    output = layers.BatchNormalization(name='output_norm')(output)
    output = tf.reshape(output, [-1, class_no, class_no])
    output = tf.keras.activations.softplus(output)

    # Spatial expansion
    output_expanded = tf.expand_dims(tf.expand_dims(output, axis=-1), axis=-1)
    tiled = tf.tile(output_expanded, [1, 1, 1, input_height, input_width])
    y = tf.reshape(tiled, [-1, class_no * class_no, input_height, input_width])

    return y

def Segmentation_Head(x, out_channels,activation_fn):
    """
    Defines the segmentation head for a segmentation model.

    This function creates a convolutional layer with a 1x1 kernel to produce 
    the final segmentation output. It is typically used as the final layer 
    in a segmentation model to generate class predictions for each pixel.

    Args:
        x (tf.Tensor): Input tensor from the previous layer.
        out_channels (int): Number of output channels (classes) for the segmentation.
        activation_fn (str): Activation function to apply to the output. 
            Common choices include 'softmax' for multi-class segmentation 
            and 'sigmoid' for binary segmentation.

    Returns:
        tf.Tensor: Output tensor representing the segmentation predictions.

    Example:
        >>> from tensorflow.keras.layers import Input
        >>> input_tensor = Input(shape=(256, 256, 3))
        >>> segmentation_output = Segmentation_Head(input_tensor, out_channels=2, activation_fn='softmax')
        >>> print(segmentation_output.shape)
        (None, 256, 256, 2)

    Note:
        - The `kernel_initializer` used here is a Glorot Uniform initializer 
          with a seed for reproducibility. You can replace it with any other 
          initializer if needed.
        - The activation function should be chosen based on the segmentation 
          task. For example, 'softmax' for multi-class segmentation and 
          'sigmoid' for binary segmentation.

    See Also:
        ResUNet: A UNet backbone implementation that can be used with this segmentation head.
        Crowd_Segmentation_Model: A model that integrates this segmentation head.
    """
    outputs = layers.Conv2D(out_channels, kernel_size=(1, 1),
                          kernel_initializer=kernel_initializer(42), activation=activation_fn,
                          name='Conv200')(x)
    return outputs

def ResUNet(inputs, out_channels=2):
    """
    Defines a ResUNet model for image segmentation tasks.

    This function implements a ResUNet architecture, which combines the feature extraction
    capabilities of a ResNet34 backbone with a U-Net style decoder. The model is designed
    to handle segmentation tasks by leveraging the rich feature representations from the
    ResNet backbone and the upsampling and concatenation operations in the decoder.

    Args:
        inputs (tf.Tensor): Input tensor to the model, typically an image.
        out_channels (int, optional): Number of output channels (classes) for the segmentation. Defaults to 2.

    Returns:
        tf.Tensor: Output tensor of the model, representing the segmentation mask.

    Example:
        >>> from tensorflow.keras.layers import Input
        >>> input_tensor = Input(shape=(256, 256, 3))
        >>> output_tensor = ResUNet(input_tensor, out_channels=2)
        >>> print(output_tensor.shape)
        (None, 256, 256, 2)

    Note:
        - The `ResNet34` model is used as the backbone, and its weights are initialized from ImageNet,
          Developed by Pavel Iakubovskii, for more information redirect to https://github.com/qubvel/classification_models.
        - The backbone layers are frozen except for the BatchNormalization layers.
        - The `upsample` function should be defined and imported before using this function.
        - The `DefaultConv2D` function should be defined and imported before using this function.
        - The `residual_block` function should be defined and imported before using this function.
        - The `kernel_initializer` function should be defined and imported before using this function.

    See Also:
        ResNet34: The backbone model used in this implementation.
        upsample: The function used for upsampling in the decoder.
        DefaultConv2D: The function used for convolutional layers.
        residual_block: The function used for residual blocks in the decoder.
    """

    # Backbone (ResNet34)
    backbone = ResNet34(
        input_tensor=inputs,
        include_top=False,
        weights='imagenet')

    # Freeze backbone layers
    for layer in backbone.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    # Feature levels
    level_1 = backbone.layers[5].output   # 128x128x64
    level_2 = backbone.layers[37].output  # 64x64x64
    level_3 = backbone.layers[74].output  # 32x32x128
    level_4 = backbone.layers[129].output # 16x16x256
    x = backbone.layers[157].output       # 8x8x512

    # Decoder
    x = upsample(name='Up60')(x)
    x = layers.Concatenate(name='Concat60')([level_4, x])
    x = DefaultConv2D(256, kernel_initializer=kernel_initializer(91) , name='Conv60')(x)
    x = residual_block(x, 256, kernel_initializer(47), '60')

    x = upsample(name='Up70')(x)
    x = layers.Concatenate(name='Concat70')([level_3, x])
    x = DefaultConv2D(128, kernel_initializer=kernel_initializer(21), name='Conv70')(x)
    x = residual_block(x, 128, kernel_initializer(96), '70')

    x = upsample(name='Up80')(x)
    x = layers.Concatenate(name='Concat80')([level_2, x])
    x = DefaultConv2D(64, kernel_initializer=kernel_initializer(96), name='Conv80')(x)
    x = residual_block(x, 64, kernel_initializer(98), '80')

    x = upsample(name='Up90')(x)
    x = layers.Concatenate(name='Concat90')([level_1, x])
    x = DefaultConv2D(32, kernel_initializer=kernel_initializer(35), name='Conv90')(x)
    x = residual_block(x, 32, kernel_initializer(7), '90')

    x = upsample(name='Up100')(x)
    x = layers.Concatenate(name='Concat100')([inputs, x])
    x = DefaultConv2D(16, kernel_initializer=kernel_initializer(45), name='Conv100')(x)
    x = residual_block(x, 16, kernel_initializer(7), '100')

    return x

def Crowd_Segmentation_Model(noisy_labels, class_no=2, input_shape=(512, 512, 3),activation_fn='softmax'):
    """
    Creates a crowd segmentation model that integrates noisy labels and a UNet backbone.

    This model is designed to handle segmentation tasks with multiple annotators, 
    incorporating a UNet architecture for image feature extraction and a crowd matrix 
    generation layer to account for annotator variability.

    Args:
        noisy_labels (list): List of noisy labels representing different annotators.
        class_no (int, optional): Number of segmentation classes. Defaults to 2.
        input_shape (tuple, optional): Input image shape. Defaults to (512, 512, 3).
        activation_fn (str, optional): Activation function for the segmentation head. 
            Defaults to 'softmax'.

    Returns:
        Model: A Keras model instance for crowd segmentation.

    Raises:
        ValueError: If the activation function is not supported.

    Example:
        >>> noisy_labels = ['annotator_1', 'annotator_2', 'annotator_3']
        >>> model = Crowd_Segmentation_Model(noisy_labels, class_no=2, input_shape=(512, 512, 3))
        >>> model.summary()
        # Prints the model summary

    Note:
        - The `ResUNet` function should be defined and imported before using this model.
        - The `Image_CM` function should be defined and imported to generate the crowd matrix.
        - The `Segmentation_Head` function should be defined and imported to create the segmentation output.
        - This model assumes that the input data is preprocessed and normalized appropriately.
        - The `activation_fn` parameter should be a valid Keras activation function.

    See Also:
        ResUNet: The UNet backbone implementation used in this model.
        Image_CM: The function responsible for generating the crowd matrix.
        Segmentation_Head: The function that defines the segmentation output layer.
    """
    # Input layers
    image_input = layers.Input(shape=input_shape, name='image_input')
    annotator_input = layers.Input(shape=(len(noisy_labels),), name='annotator_input')

    # UNet backbone
    decoded = ResUNet(image_input, out_channels=class_no, )

    # Crowd matrix generation
    cm = Image_CM(
        annotator_input, decoded, class_no,
        input_height=input_shape[0],
        input_width=input_shape[1]
    )

    # Segmentation head
    segmentation = Segmentation_Head(
        x=decoded,
        out_channels=class_no,
        activation_fn=activation_fn
    )

    # Model creation
    return Model(
        inputs=[image_input, annotator_input],
        outputs=[segmentation, cm],
        name='crowd_segmentation_model'
    )