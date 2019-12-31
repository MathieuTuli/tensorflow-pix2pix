from pathlib import Path
import tensorflow as tf
import os

from .components import ImageDirection
from .network.train import fit
from .helpers import (
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

    PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')
    inp, real = load(Path(PATH), ImageDirection.AtoB)
