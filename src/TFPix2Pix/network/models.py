from tensorflow.keras import layers, Model
from typing import Union, Tuple, List
from pathlib import Path

import tensorflow as tf
import logging

# from .layers import (
#     Downsample,
#     Upsample)
from .layers import (
    downsample_sequential as Downsample,
    upsample_sequential as Upsample)
from .helpers import load_image


def generator(input_shape: Tuple[int, int, int]):
    """
    """

    inputs = tf.keras.layers.Input(shape=input_shape)
    down_stack = [
        Downsample(filters=64, kernel_size=4,
                   batch_norm=False),  # , input_shape=input_shape),
        Downsample(filters=128, kernel_size=4),
        Downsample(filters=256, kernel_size=4),
        Downsample(filters=512, kernel_size=4),
        Downsample(filters=512, kernel_size=4),
        Downsample(filters=512, kernel_size=4),
        Downsample(filters=512, kernel_size=4),
        Downsample(filters=512, kernel_size=4)]
    up_stack = [
        Upsample(filters=512, kernel_size=4, dropout=True),
        Upsample(filters=512, kernel_size=4, dropout=True),
        Upsample(filters=512, kernel_size=4, dropout=True),
        Upsample(filters=512, kernel_size=4),
        Upsample(filters=256, kernel_size=4),
        Upsample(filters=128, kernel_size=4),
        Upsample(filters=64, kernel_size=4)]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(
        filters=input_shape[2],
        kernel_size=4,
        strides=2,
        padding='same',
        kernel_initializer=initializer,
        activation='tanh')
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def generator_loss(disc_generated_output, gen_output,
                   target, loss_object, _lambda):
    gan_loss = loss_object(tf.ones_like(
        disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (_lambda * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output, loss_object):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(
        disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


class Generator(Model):
    """
    """

    def __init__(self,
                 output_channels: int,
                 input_shape: Tuple[int, int, int] = None,
                 **kwargs) -> None:
        '''
        '''
        super(Generator, self).__init__(**kwargs)
        self.down_stack = [
            Downsample(filters=64, kernel_size=4,
                       batch_norm=False),  # , input_shape=input_shape),
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


def discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = Downsample(filters=64, kernel_size=4, batch_norm=False)(x)
    down2 = Downsample(filters=128, kernel_size=4)(down1)
    down3 = Downsample(filters=256, kernel_size=4)(down2)
    zero_pad1 = layers.ZeroPadding2D()(down3)
    down4 = Downsample(filters=512, kernel_size=4, strides=1)(zero_pad1)
    zero_pad2 = layers.ZeroPadding2D()(down4)
    last = layers.Conv2D(filters=1, kernel_size=4, strides=1,
                         kernel_initializer=initializer)(zero_pad2)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)


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
    def loss(
            real_output: tf.keras.Model,
            generated_output: tf.keras.Model,) -> tf.losses.BinaryCrossentropy:
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = loss_object(tf.ones_like(real_output), real_output)
        generated_loss = loss_object(
            tf.zeros_like(generated_output), generated_output)
        total_loss = real_loss + generated_loss
        return total_loss
