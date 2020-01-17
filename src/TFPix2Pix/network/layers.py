from tensorflow.keras import layers
from typing import Union, Tuple, List

import tensorflow as tf
import logging


class Downsample(layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 batch_norm: bool = True,
                 strides: Union[int, Tuple[int, int]] = 2,
                 input_shape: Tuple[int, int, int] = None,
                 **kwargs) -> None:
        super(Downsample, self).__init__(**kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)
        if input_shape is not None:
            self.layer = layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                kernel_initializer=initializer,
                input_shape=input_shape,
                use_bias=False)
        else:
            self.layer = layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                kernel_initializer=initializer,
                use_bias=False)
        self.batch_norm = None if not batch_norm else \
            layers.BatchNormalization()
        self.activation = layers.LeakyReLU()

    def call(self,
             inputs: Union[tf.Tensor,
                           Tuple[tf.Tensor, ...],
                           List[tf.Tensor]],
             **kwargs) -> Union[tf.Tensor,
                                Tuple[tf.Tensor, ...],
                                List[tf.Tensor]]:
        layer = self.layer(inputs)
        if self.batch_norm is not None:
            layer = self.batch_norm(layer)
        layer = self.activation(layer)
        return layer


class Upsample(layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 dropout: bool = False,
                 **kwargs) -> None:
        super(Upsample, self).__init__(**kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)
        self.layer = layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False)
        self.batch_norm = layers.BatchNormalization()
        self.dropout = None if not dropout else layers.Dropout(0.5)
        self.activation = layers.LeakyReLU()

    def call(self,
             inputs: Union[tf.Tensor,
                           Tuple[tf.Tensor, ...],
                           List[tf.Tensor]],
             **kwargs) -> Union[tf.Tensor,
                                Tuple[tf.Tensor, ...],
                                List[tf.Tensor]]:
        layer = self.layer(inputs)
        layer = self.batch_norm(layer)
        if self.dropout is not None:
            layer = self.dropout(layer)
        layer = self.activation(layer)
        return layer
