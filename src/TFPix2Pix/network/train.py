from IPython import display

import tensorflow as tf
import time

from .models import Generator, Discriminator
from .helpers import (
    load,
    resize,
    random_crop,
    normalize,
    random_jitter,
    load_image_train,
    load_image_test,)

display.IFrame(
    src="https://tensorboard.dev/experiment/lZ0C6FONROaUMfjYkVyJqw",
    width="100%",
    height="1000px")


@tf.function
def fit(train_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
        checkpoint_path: str,
        epochs: int = 150,
        batch_size: int = 1) -> None:
    generator = Generator(output_channels=3)
    print("TEST\n\n\n")
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

    @tf.function
    def train_step(
            input_image: tf.data.Dataset,
            target: tf.data.Dataset,
            epoch: int) -> None:
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator(
                [input_image, target], training=True)
            disc_generated_output = discriminator(
                [input_image, gen_output], training=True)

            gen_loss, gan_loss, gen_l1_loss = Generator.loss(
                gen_output, disc_generated_output, target, _lambda=100)
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
            tf.summary.scalar('gen_total_loss',
                              gen_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss',
                              gen_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss',
                              gen_l1_loss, step=epoch)
            tf.summary.scalar('disc_loss',
                              disc_loss, step=epoch)

    import datetime
    log_dir = "logs/"

    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    for epoch in range(epochs):
        start = time.time()

        # Train
        for input_image, target in train_ds:
            train_step(input_image, target, epoch)

        display.clear_output(wait=True)
        # Test on the same image so that the progress of the model can be
        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_path)

        print(
            'Time taken for epoch {} is {} sec\n'.format(
                epoch + 1,
                time.time()-start))
