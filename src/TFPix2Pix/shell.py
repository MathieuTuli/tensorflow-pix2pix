from pathlib import Path
import tensorflow as tf
import os

from .components import ImageDirection
from .network.train import fit
from .network.helpers import (
    load,
    resize,
    random_crop,
    normalize,
    random_jitter,
    load_image_train,
    load_image_test,)


def main():
    URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

    path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                          origin=URL,
                                          extract=True)

    BUFFER_SIZE = 400
    BATCH_SIZE = 1
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    LAMBDA = 100
    checkpoint_dir = './training_checkpoints'
    checkpoint_path = os.path.join(checkpoint_dir, "ckpt")
    PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')
    train_dataset = tf.data.Dataset.list_files(PATH + 'train/*.jpg')
    train_dataset = train_dataset.map(
        load_image_train,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    fit(train_dataset, test_dataset, checkpoint_path, epochs=150)
