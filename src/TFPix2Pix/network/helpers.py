'''
UHINET Network Pix2Pix Data Helpers

@credit: Google (https://www.tensorflow.org/tutorials/generative/pix2pix)
'''
from typing import Tuple

import tensorflow as tf

from ..components import ImageDirection


def load(image_path: str,
         direction: ImageDirection,
         input_shape: Tuple[int, int, int]) -> Tuple[tf.Tensor, tf.Tensor]:
    '''
    @returns: Tuple, (input_image, target_image)
    '''

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=input_shape[2])

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    target_image = tf.cast(real_image, tf.float32)

    if direction == ImageDirection.BtoA:
        return input_image, target_image
    else:
        return target_image, input_image


def resize(input_image: tf.Tensor,
           real_image: tf.Tensor,
           height: int, width: int) -> Tuple[tf.Tensor, tf.Tensor]:
    input_image = tf.image.resize(
        input_image, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(
        real_image, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def random_crop(input_image: tf.Tensor,
                real_image: tf.Tensor,
                height: int,
                width: int,
                channels: int) -> Tuple[tf.Tensor, tf.Tensor]:
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, height, width, channels])

    return cropped_image[0], cropped_image[1]


def normalize(input_image: tf.Tensor,
              real_image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image: tf.Tensor,
                  real_image: tf.Tensor,
                  height: int,
                  width: int,
                  channels: int) -> Tuple[tf.Tensor, tf.Tensor]:
    # Random jittering as described in the paper is to
    # 1. Resize an image to bigger height and width
    # 2. Randomnly crop to the original size
    # 3. Randomnly flip the image horizontally
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(
        input_image, real_image, height, width, channels)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(
        image_path: str,
        image_direction: ImageDirection,
        input_shape: Tuple[int, int, int]) -> Tuple[tf.Tensor, tf.Tensor]:
    height = input_shape[0]
    width = input_shape[1]
    channels = input_shape[2]
    input_image, real_image = load(image_path, image_direction, input_shape)
    input_image, real_image = random_jitter(
        input_image, real_image, height, width, channels)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(
        image_path: str,
        image_direction: ImageDirection,
        input_shape: Tuple[int, int, int]) -> Tuple[tf.Tensor, tf.Tensor]:
    height = input_shape[0]
    width = input_shape[1]
    input_image, real_image = load(image_path, image_direction, input_shape)
    input_image, real_image = resize(input_image, real_image,
                                     height, width)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image(image_path: str,
               input_shape: Tuple[int, int, int]) -> tf.Tensor:
    height = input_shape[0]
    width = input_shape[1]
    channels = input_shape[2]
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=channels)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(
        image, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image = (image / 127.5) - 1
    image = tf.expand_dims(image, axis=0)
    return image
