import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers, regularizers

def display_learning_curves(history,NUM_EPOCHS):
    """Display the learning curves of training and validation loss over epochs.

    This function plots the training and validation loss over the specified number of epochs
    to visualize the learning progress of a neural network model.

    Parameters:
        history (History): A History object containing the training history of the model.
        NUM_EPOCHS (int): The total number of epochs for which the model was trained.

    Returns:
        None

    """
    # Extracting loss values
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    # Generating epoch range
    epochs_range = range(NUM_EPOCHS)  # Assuming NUM_EPOCHS is defined

    # Creating figure
    fig = plt.figure(figsize=(12, 6))

    # Plotting loss curves
    plt.plot(epochs_range, loss, label="train loss")
    plt.plot(epochs_range, val_loss, label="validation loss")

    # Adding titles and labels
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    # Adjusting layout
    fig.tight_layout()

    # Displaying grid and plot
    plt.grid()
    plt.show()

def create_mask(pred_mask):
    """Creates a segmentation mask from the predicted mask tensor.

    Parameters:
        pred_mask (tensor): Predicted mask tensor with shape (height, width, num_classes).

    Returns:
        A segmentation mask tensor with shape (height, width, 1), where each pixel 
        represents the predicted class index.

    """
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def display(display_list, dice):
    """Displays images and masks in a row with titles.

    This function creates a visualization of images and masks in a row, with titles 
    for each component. It is commonly used to visualize input images, true masks, 
    and predicted masks in semantic segmentation tasks.

    Parameters:
        display_list (list): A list containing the images/masks to be displayed.
        dice (float): The Dice coefficient value to be included in the title of 
            the predicted mask image.

    Returns:
        None

    """
    plt.figure(figsize=(15, 15))
    title = ["Input Image", "True Mask", f"Predicted Mask (dice={dice:.3f})"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis("off")
    plt.show()

def show_predictions(model, dataset, num=1):
    """Displays predictions of a model on a dataset along with ground truth masks.

    This function takes a trained model and a dataset containing input images and 
    corresponding ground truth masks, generates predictions using the model, and 
    displays the original image, ground truth mask, and predicted mask for visualization.

    Parameters:
        model (tensorflow.keras.Model): A trained segmentation model.
        dataset (tensorflow.data.Dataset): A dataset containing input images and 
            corresponding ground truth masks.
        num (int): Number of samples from the dataset to visualize. Defaults to 1.

    Returns:
        None

    """
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        # Assuming dice_coef.call is an instance of DiceCoefficient class
        dice = dice_coef.call(mask, pred_mask)
        display([image[0], mask[0], pred_mask[0]], np.abs(dice[0]))


