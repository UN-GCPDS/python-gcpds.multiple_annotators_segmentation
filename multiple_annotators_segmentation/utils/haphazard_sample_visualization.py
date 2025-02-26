import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def haphazard_sample_visualization(partition_dataset, num_classes, num_annotators, original_masks=True):
    print("Haphazard sample visualization")