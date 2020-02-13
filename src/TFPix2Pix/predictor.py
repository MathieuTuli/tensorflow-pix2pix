from typing import Tuple
from pathlib import Path

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
        image, _ = load_image_test(
            str(image_path), ImageDirection.AtoB, self.input_shape)
        prediction = self.generator(image, training=True)
        return np.array(prediction[0], dtype=np.uint8)
