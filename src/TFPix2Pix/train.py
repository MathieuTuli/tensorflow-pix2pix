from argparse import _SubParsersAction
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


def control_c_handler(_signal, frame):
    print("\n---------------------------------")
    print("Pix2Pix Training: Ctrl-C. Shutting Down.")
    print("---------------------------------")
    sys.exit(0)


@tf.function
def fit(dataset_path: str,
        checkpoint_path: str,
        image_direction: ImageDirection,
        epochs: int,
        batch_size: int,
        buffer_size: int,
        _lambda: int,
        checkpoint_save_freq: int,
        gpu: bool,) -> None:
    """
    @param checkpoint_save_freq: int: number of epochs before saving checkpoint
                                      ie. checkpoint_save_freq = 20 means
                                      saving checkpoint every 20 epochs
    """
    signal.signal(signal.SIGINT, control_c_handler)
    print("\n---------------------------------")
    print("TFPix2Pix Training")
    print("---------------------------------\n")

    train_dataset = tf.data.Dataset.list_files(dataset_path + 'train/*')
    train_dataset = train_dataset.map(
        load_image_train,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = tf.data.Dataset.list_files(dataset_path + 'test/*')
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(batch_size)

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
                f"TFPix2Pix Train: Image: {n} / {len(train_dataset)}")
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_output = generator(input_image, training=True)

                disc_real_output = discriminator(
                    [input_image, target], training=True)
                disc_generated_output = discriminator(
                    [input_image, gen_output], training=True)

                gen_loss, gan_loss, gen_l1_loss = Generator.loss(
                    gen_output, disc_generated_output, target, _lambda=_lambda)
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

        if (epoch + 1) % checkpoint_save_freq == 0:
            checkpoint.write(file_prefix=checkpoint_path)

        end = time.time()
        logging.info(
            'TFPix2Pix Train: Time taken for epoch {} is {} sec\n'.format(
                epoch + 1,
                end - start))
        logging.info(
            "TFPix2Pix Train: Estimated time remaining: " +
            "{(epochs - epoch + 1) * (end - start))}s")
    return True
