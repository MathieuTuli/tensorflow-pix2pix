from argparse import _SubParsersAction
from pathlib import Path
from typing import Tuple

from matplotlib import pyplot as plt

import tensorflow as tf
import subprocess
import traceback
import datetime
import logging
import signal
import time
import sys

from .network.helpers import load_image_train, load_image_test
from .network.models import Generator, Discriminator
from .components import ImageDirection
from .helpers import generate_images


def args(sub_parser: _SubParsersAction) -> None:
    sub_parser.add_argument('--data', type=str,
                            dest='data',
                            required=True,
                            help='Required. Dataset path')
    sub_parser.add_argument('--checkpoint', type=str,
                            dest='checkpoint',
                            required=True,
                            help='Required. Checkpoint path')
    sub_parser.add_argument('--log-dir', type=str,
                            dest='log_dir',
                            required=False,
                            default=None,
                            help='Default = checkpoint dir. Log dir path. ' +
                            'logs will be written to "args.checkpoint/logs"')
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
    sub_parser.add_argument('--input-shape', type=list,
                            dest='input_shape',
                            default=(256, 256, 3),
                            help='Default = (256, 256, 3). Input Shape.')
    sub_parser.add_argument(
        '--gpu', action='store_true',
        dest='gpu',
        help="Default = False. Set if using gpu")
    sub_parser.set_defaults(gpu=False)
    sub_parser.add_argument(
        '--tensorboard', action='store_true',
        dest='tensorboard',
        help="Default = False. Set if using tensorboard")
    sub_parser.set_defaults(gpu=False)
    sub_parser.add_argument(
        '--eager', action='store_true',
        dest='eager',
        help="Default = False. Set if using eager execution")
    sub_parser.set_defaults(eager=False)


def control_c_handler(_signal, frame):
    print("\n---------------------------------")
    print("Pix2Pix Train: Ctrl-C. Shutting Down.")
    print("---------------------------------")
    sys.exit(0)


# @tf.function
def fit(dataset_path: Path,
        checkpoint_path: Path,
        log_dir: Path,
        image_direction: ImageDirection,
        epochs: int,
        batch_size: int,
        buffer_size: int,
        _lambda: int,
        checkpoint_save_freq: int,
        gpu: bool,
        tensorboard: bool,
        eager: bool,
        input_shape: Tuple[int, int, int]) -> None:
    """
    @param checkpoint_save_freq: int: number of epochs before saving checkpoint
                                      ie. checkpoint_save_freq = 20 means
                                      saving checkpoint every 20 epochs
    """
    signal.signal(signal.SIGINT, control_c_handler)
    print("\n---------------------------------")
    print("TFPix2Pix Train")
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
                  f"@param gpu                   |  {gpu} \n"
                  f"@param tensorboard           |  {tensorboard} \n"
                  f"@param eager                 |  {eager} \n"
                  f"@param input_shape           |  {input_shape}")

    device = '/device:CPU:0' if not gpu else '/device:GPU:0'

    # NOTE : Hack for RTX*** GPU devices
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for dev in gpu_devices:
        tf.config.experimental.set_memory_growth(dev, True)
    train_dataset = tf.data.Dataset.list_files(
        str(dataset_path / 'train/*'))
    train_dataset = train_dataset.map(
        lambda x: load_image_train(x, image_direction, input_shape),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = tf.data.Dataset.list_files(
        str(dataset_path / 'test/*'))
    test_dataset = test_dataset.map(
        lambda x: load_image_test(x, image_direction, input_shape))
    test_dataset = test_dataset.batch(batch_size)
    process = None
    if tensorboard:
        process = subprocess.Popen([
            "tensorboard", f"--logdir={str(log_dir)}", "--host",
            "localhost", "--port", "8088"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(
            "TFPix2Pix: Train: Tensorboard started at http://localhost:8088")
        # output, error = process.communicate()
        # if output:
        #     logging.info("TFPix2Pix: Train: Tensorboard output \b{output}")
        # if error:
        #     logging.critical("TFPix2Pix: Train: Tensorboard could not start." +
        #                      " Reason: \n {error}")

    try:
        with tf.device(device):
            generator = Generator(
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
                generator=generator)
            checkpoint.restore(tf.train.latest_checkpoint(
                checkpoint_path)).expect_partial()
            checkpoint_path = checkpoint_path / 'ckpt'
            # tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
            # tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)
            summary_writer = tf.summary.create_file_writer(str(
                log_dir / "fit" /
                datetime.datetime.now().strftime("Y%m%d-%H%M%S")))

            for epoch in range(epochs):
                if epoch == 0:
                    for input_image, target in test_dataset.take(1):
                        generate_images(generator, input_image, target)
                        break
                start = time.time()
                # Train
                logging.info(f"TFPix2Pix Train: Epoch: {epoch + 1} / {epochs}")
                for n, (input_image, target) in train_dataset.enumerate():
                    logging.debug(
                        f"TFPix2Pix Train: Image: {n}")

                    # TODO: Why does adding tf.function() make it run slower
                    # @tf.function()
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
                        with summary_writer.as_default():
                            tf.summary.scalar(
                                'gen_total_loss', gen_loss, step=epoch)
                            tf.summary.scalar(
                                'gen_gan_loss', gan_loss, step=epoch)
                            tf.summary.scalar(
                                'gen_l1_loss', gen_l1_loss, step=epoch)
                            tf.summary.scalar(
                                'disc_loss', disc_loss, step=epoch)
                    train_step(input_image=input_image, target=target)

                if (epoch + 1) % checkpoint_save_freq == 0:
                    checkpoint.save(file_prefix=str(checkpoint_path))
                    # generator.save_weights(
                    #     str(checkpoint_path / 'generator.ckpt'))
                    logging.info("TFPix2Pix Train: checkpoint saved.")

                end = time.time()
                logging.info(
                    'TFPix2Pix Train: Time taken for epoch ' +
                    f'{epoch + 1} is {end - start}s')
                logging.info(
                    "TFPix2Pix Train: Estimated time remaining: " +
                    f"{(epochs - epoch + 1) * (end - start)}s")

            # generator.save_weights(
            #     str(checkpoint_path / 'generator.ckpt'))
            for input_image, target in test_dataset.take(1):
                generate_images(generator, input_image, target)
                break
            checkpoint.save(file_prefix=str(checkpoint_path))
            # tf.saved_model.save(generator, str(checkpoint / 'generator'))
            # tf.saved_model.save(discriminator, str(
            #     discriminator / 'generator'))
    except Exception as e:
        logging.error(f"TFPix2Pix Train: {e}")
        traceback.print_exc()
        # TODO proper process cleanup
    if process is not None:
        logging.info(
            "TFPix2Pix Train: Tensorboard is still running. " +
            "Hit CTRL-C to stop it")
        process.wait()
        return False
    if process is not None:
        logging.info("TFPix2Pix: Train: Tensorboard is still running. " +
                     "Hit CTRL-C to stop it")
        process.wait()
    return True
