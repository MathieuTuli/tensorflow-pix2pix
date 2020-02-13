from typing import Tuple
from pathlib import Path

from matplotlib import pyplot as plt

import tensorflow as tf
import numpy as np


def file_exists(path: Path) -> bool:
    return path.exists()


def generate_images(model: tf.keras.Model,
                    test_input: tf.Tensor,
                    target: tf.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    # display_list = [test_input[0], target[0], prediction[0]]
    # title = ['Input Image', 'Ground Truth', 'Predicted Image']

    # for i in range(3):
    #     plt.subplot(1, 3, i+1)
    #     plt.title(title[i])
    #     # getting the pixel values between [0, 1] to plot it.
    #     plt.imshow(display_list[i] * 0.5 + 0.5)
    #     plt.axis('off')
    # plt.show()
    return target[0], prediction[0]
