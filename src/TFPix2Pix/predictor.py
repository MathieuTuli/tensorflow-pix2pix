from typing import Tuple
from pathlib import Path

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from .network.helpers import load_image, load_image_test
from .network.models import Generator, Discriminator
from .components import ImageDirection


class Predictor():
    def __init__(self,
                 weights: Path,
                 input_shape: Tuple[int, int, int]) -> None:
        self.input_shape = input_shape
        self.generator = Generator(
            output_channels=input_shape[2], input_shape=input_shape)
        generator_optimizer = tf.keras.optimizers.Adam(2e-4,
                                                       beta_1=0.5)
        discriminator = Discriminator()
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4,
                                                           beta_1=0.5)
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            discriminator=discriminator,
            generator=self.generator)
        checkpoint.restore(tf.train.latest_checkpoint(
            str(weights))).expect_partial()

    def predict(self, image_path: Path) -> np.ndarray:
        test_dataset = tf.data.Dataset.list_files(
            str(image_path))
        test_dataset = test_dataset.map(
            lambda x: load_image_test(x, ImageDirection.AtoB,
                                      self.input_shape))
        test_dataset = test_dataset.batch(1)
        for image, _ in test_dataset.take(1):
            prediction = self.generator(image, training=True)
            plt.figure(figsize=(15, 15))

            display_list = [image[0], image[0], prediction[0]]
            title = ['Input Image', 'Ground Truth', 'Predicted Image']

            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.title(title[i])
                # getting the pixel values between [0, 1] to plot it.
                plt.imshow(display_list[i] * 0.5 + 0.5)
                plt.axis('off')
            plt.show()
            return prediction[0]
