from argparse import _SubParsersAction
from pathlib import Path
import tensorflow as tf
import logging
import signal
import time
import sys

from .network.helpers import load_image_train, load_image_test
from .network.models import Generator, Discriminator
from .components import ImageDirection


def args(sub_parser: _SubParsersAction) -> None:
    sub_parser.add_argument('--data', type=str.lower,
                            dest='data',
                            required=True,
                            help='Required. Dataset path')
    sub_parser.add_argument('--checkpoint', type=str.lower,
                            dest='checkpoint',
                            required=True,
                            help='Required. Checkpoint path')
    sub_parser.add_argument(
        '--image-direction', type=ImageDirection.__getitem__,
        choices=ImageDirection.__members__.values(),
        dest='image_direction',
        required=True,
        help="Required. Image Direction")
    sub_parser.add_argument('--epochs', type=int,
                            required=True,
                            help='Required. Number of epochs to train for')
    sub_parser.add_argument('--batch-size', type=int,
                            dest='batch_size',
                            default=1,
                            help='Default = 1. Batch Size for Training.')
    sub_parser.add_argument('--buffer-size', type=int,
                            dest='buffer_size',
                            default=400,
                            help='Default = 400. Buffer Size for Training')
    sub_parser.add_argument('--lambda', type=int,
                            dest='_lambda',
                            default=100,
                            help='Default = 100. Lambda value for Training')
    sub_parser.add_argument('--save-freq', type=int,
                            dest='save_freq',
                            default=20,
                            help='Default = 20. Save every X number of epochs')
    sub_parser.add_argument(
        '--gpu', action='store_true',
        dest='gpu',
        help="Default = False. Set if using gpu")
    sub_parser.set_defaults(gpu=False)
    sub_parser.add_argument(
        '--eager', action='store_true',
        dest='eager',
        help="Default = False. Set if using eager execution")
    sub_parser.set_defaults(eager=False)


def control_c_handler(_signal, frame):
    print("\n---------------------------------")
    print("Pix2Pix Training: Ctrl-C. Shutting Down.")
    print("---------------------------------")
    sys.exit(0)


# @tf.function
def fit(dataset_path: Path,
        checkpoint_path: Path,
        image_direction: ImageDirection,
        epochs: int,
        batch_size: int,
        buffer_size: int,
        _lambda: int,
        checkpoint_save_freq: int,
        gpu: bool,
        eager: bool) -> None:
    """
    @param checkpoint_save_freq: int: number of epochs before saving checkpoint
                                      ie. checkpoint_save_freq = 20 means
                                      saving checkpoint every 20 epochs
    """
    signal.signal(signal.SIGINT, control_c_handler)
    print("\n---------------------------------")
    print("TFPix2Pix Training")
    print("---------------------------------\n")
    logging.debug("TFPix2Pix Train: Fit arguments: \n"
                  f"@param dataset_path          |  {dataset_path} \n"
                  f"@param checkpoint_path       |  {checkpoint_path} \n"
                  f"@param image_direction       |  {image_direction} \n"
                  f"@param epochs                |  {epochs} \n"
                  f"@param batch_size            |  {batch_size} \n"
                  f"@param buffer_size           |  {buffer_size} \n"
                  f"@param _lambda               |  {_lambda} \n"
                  f"@param checkpoint_save_freq  |  {checkpoint_save_freq} \n"
                  f"@param gpu                   |  {gpu} \n")

    device = '/device:CPU:0' if not gpu else '/device:GPU:0'
    train_dataset = tf.data.Dataset.list_files(
        str(dataset_path / 'train/*'))
    train_dataset = train_dataset.map(
        load_image_train,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = tf.data.Dataset.list_files(
        str(dataset_path / 'test/*'))
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(batch_size)

    try:
        with tf.device(device):
            generator = Generator(output_channels=3)
            generator_optimizer = tf.keras.optimizers.Adam(2e-4,
                                                           beta_1=0.5)
            discriminator = Discriminator()
            discriminator_optimizer = tf.keras.optimizers.Adam(2e-4,
                                                               beta_1=0.5)
            checkpoint = tf.train.Checkpoint(
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                generator=generator,
                discriminator=discriminator)

            for epoch in range(epochs):
                start = time.time()

                # Train
                logging.info(f"TFPix2Pix Train: Epoch: {epoch} / {epochs}")
                for n, (input_image, target) in train_dataset.enumerate():
                    logging.debug(
                        f"TFPix2Pix Train: Image: {n}")

                    # TODO: Why does adding tf.function() make it run slower
                    @tf.function()
                    def train_step(input_image: tf.Tensor,
                                   target: tf.Tensor) -> None:
                        with tf.GradientTape() as gen_tape, tf.GradientTape() \
                                as disc_tape:
                            gen_output = generator(input_image, training=True)

                            disc_real_output = discriminator(
                                [input_image, target], training=True)
                            disc_generated_output = discriminator(
                                [input_image, gen_output], training=True)

                            gen_loss, gan_loss, gen_l1_loss = Generator.loss(
                                gen_output, disc_generated_output,
                                target, _lambda=_lambda)
                            disc_loss = Discriminator.loss(
                                disc_real_output, disc_generated_output)

                        generator_gradients = gen_tape.gradient(
                            gen_loss,
                            generator.trainable_variables)
                        discriminator_gradients = disc_tape.gradient(
                            disc_loss,
                            discriminator.trainable_variables)

                        generator_optimizer.apply_gradients(
                            zip(generator_gradients,
                                generator.trainable_variables))
                        discriminator_optimizer.apply_gradients(
                            zip(discriminator_gradients,
                                discriminator.trainable_variables))
                    train_step(input_image=input_image, target=target)

                if (epoch + 1) % checkpoint_save_freq == 0:
                    checkpoint.write(file_prefix=checkpoint_path)

                end = time.time()
                logging.info(
                    'TFPix2Pix Train: Time taken for epoch ' +
                    f'{epoch + 1} is {end - start}s')
                logging.info(
                    "TFPix2Pix Train: Estimated time remaining: " +
                    f"{(epochs - epoch + 1) * (end - start)}s")
    except Exception as e:
        logging.error(f"TFPix2Pix Train: {e}")
        return False
    return True
