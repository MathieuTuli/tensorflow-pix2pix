from tensorflow.keras import layers, Model
from typing import Union, Tuple, List
from pathlib import Path

import tensorflow as tf
import logging

from .layers import (
    Downsample,
    Upsample)
from .helpers import load_image


class Generator(Model):
    """
    """

    def __init__(self,
                 output_channels: int,
                 **kwargs) -> None:
        '''
        '''
        super(Generator, self).__init__(**kwargs)
        self.down_stack = [
            Downsample(filters=64, kernel_size=4,
                       batch_norm=False),
            Downsample(filters=128, kernel_size=4),
            Downsample(filters=256, kernel_size=4),
            Downsample(filters=512, kernel_size=4),
            Downsample(filters=512, kernel_size=4),
            Downsample(filters=512, kernel_size=4),
            Downsample(filters=512, kernel_size=4),
            Downsample(filters=512, kernel_size=4)]
        self.up_stack = [
            Upsample(filters=512, kernel_size=4, dropout=True),
            Upsample(filters=512, kernel_size=4, dropout=True),
            Upsample(filters=512, kernel_size=4, dropout=True),
            Upsample(filters=512, kernel_size=4),
            Upsample(filters=256, kernel_size=4),
            Upsample(filters=128, kernel_size=4),
            Upsample(filters=64, kernel_size=4)]
        initializer = tf.random_normal_initializer(0., 0.02)
        self.last = layers.Conv2DTranspose(
            filters=output_channels,
            kernel_size=4,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            activation='tanh')
        self.concat = layers.Concatenate()

    def call(self,
             inputs: Union[tf.Tensor,
                           Tuple[tf.Tensor, ...],
                           List[tf.Tensor]],
             training: bool = False) -> Union[tf.Tensor,
                                              Tuple[tf.Tensor, ...],
                                              List[tf.Tensor]]:
        layer = inputs
        skips = []
        for downsample in self.down_stack:
            layer = downsample(layer)
            skips.append(layer)
        skips = reversed(skips[:-1])
        for upsample, skip in zip(self.up_stack, skips):
            layer = upsample(layer)
            layer = self.concat([layer, skip])

        layer = self.last(layer)
        return layer

    @staticmethod
    @tf.function()
    def loss(generator_output: tf.keras.Model,
             discriminator_output: tf.keras.Model,
             target: tf.data.Dataset,
             _lambda: int) -> Tuple[
            tf.losses.BinaryCrossentropy,
            tf.losses.BinaryCrossentropy,
            tf.losses.BinaryCrossentropy]:

        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        gan_loss = loss_object(tf.ones_like(
            discriminator_output), discriminator_output)
        l1_loss = tf.reduce_mean(tf.abs(target - generator_output))
        total_gen_loss = gan_loss + (_lambda * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def infer(self, image_path: Path):
        ...


class Discriminator(Model):
    """
    """

    def __init__(self, **kwargs) -> None:
        '''
        '''
        super(Discriminator, self).__init__(**kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)
        self.concat = layers.Concatenate()
        self.stack = [
            Downsample(filters=64, kernel_size=4, batch_norm=False),
            Downsample(filters=128, kernel_size=4),
            Downsample(filters=256, kernel_size=4),
            layers.ZeroPadding2D(),
            Downsample(filters=512, kernel_size=4, strides=1),
            layers.ZeroPadding2D(),
            layers.Conv2D(filters=1, kernel_size=4, strides=1,
                          kernel_initializer=initializer)]

    def call(self,
             inputs: Union[tf.Tensor,
                           Tuple[tf.Tensor, ...],
                           List[tf.Tensor]],
             training: bool = False) -> Union[tf.Tensor,
                                              Tuple[tf.Tensor, ...],
                                              List[tf.Tensor]]:

        _input, _target = inputs
        layer = self.concat([_input, _target])
        for _layer in self.stack:
            layer = _layer(layer)
        return layer

    @staticmethod
    @tf.function()
    def loss(
            real_output: tf.keras.Model,
            generated_output: tf.keras.Model,) -> tf.losses.BinaryCrossentropy:
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = loss_object(tf.ones_like(real_output), real_output)
        generated_loss = loss_object(
            tf.zeros_like(generated_output), generated_output)
        total_loss = real_loss + generated_loss
        return total_loss
