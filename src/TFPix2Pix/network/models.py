from tensorflow.keras import layers, Model
from typing import Union, Tuple, List

import tensorflow as tf
import logging

from .data_helpers import (
    load,
    resize,
    random_crop,
    normalize,
    random_jitter,
    load_image_train,
    load_image_test,)
from .layers import (
    Downsample,
    Upsample)


class Generator(Model):
    """
    """

    def __init__(self,
                 output_channels: int,
                 **kwargs) -> None:
        '''
        '''
        super(Generator, self).__init__(**kwargs)
        self.first = Downsample(filters=64, kernel_size=4,
                                batch_norm=False)
        self.down_stack = [
            Downsample(filters=128, kernel_size=4),
            Downsample(filters=256, kernel_size=4),
            Downsample(filters=512, kernel_size=4),
            Downsample(filters=512, kernel_size=4),
            Downsample(filters=512, kernel_size=4),
            Downsample(filters=512, kernel_size=4),
            Downsample(filters=512, kernel_size=4), ]
        self.up_stack = [
            Upsample(filters=512, kernel_size=4, dropout=True),
            Upsample(filters=512, kernel_size=4, dropout=True),
            Upsample(filters=512, kernel_size=4, dropout=True),
            Upsample(filters=512, kernel_size=4),
            Upsample(filters=256, kernel_size=4),
            Upsample(filters=128, kernel_size=4),
            Upsample(filters=64, kernel_size=4), ]
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
        layer = self.first(inputs)
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
            tf.random_normal_initializer(0., 0.02),
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


class Pix2Pix(Model):
    """
    """

    def __init__(self,
                 output_channels: int,
                 **kwargs) -> None:
        '''
        '''
        super(Pix2Pix, self).__init__(**kwargs)

    def call(self,
             inputs: Union[tf.Tensor,
                           Tuple[tf.Tensor, ...],
                           List[tf.Tensor]],
             training: bool = False) -> Union[tf.Tensor,
                                              Tuple[tf.Tensor, ...],
                                              List[tf.Tensor]]:
        return
