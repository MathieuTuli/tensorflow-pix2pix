from typing import Tuple
from pathlib import Path

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from .network.helpers import load_image, load_image_test
# from .network.models import Generator, Discriminator
from .network.models import generator as Generator, \
    discriminator as Discriminator
from .components import ImageDirection


class Predictor():
    def __init__(self,
                 weights: Path,
                 input_shape: Tuple[int, int, int]) -> None:
        self.input_shape = input_shape
        # self.generator = Generator(
        #     output_channels=input_shape[2], input_shape=input_shape)
        self.generator = Generator(input_shape=input_shape)
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

    def predict(self, image: tf.Tensor) -> np.ndarray:
        prediction = self.generator(image, training=True)
        display_list = [image[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()
        plt.savefig('/home/mat/Downloads/image.png')
        return (prediction[0] * 0.5 + 0.5).numpy()
