import tensorflow as tf
import time
import os

from .models import Generator, Discriminator
from .helpers import (
    load,
    resize,
    random_crop,
    normalize,
    random_jitter,
    load_image_train,
    load_image_test,)


@tf.function
def fit(dataset_path: str,
        checkpoint_path: str,
        epochs: int = 30,
        batch_size: int = 1) -> bool:
    URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

    path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                          origin=URL,
                                          extract=True)

    BUFFER_SIZE = 400
    BATCH_SIZE = batch_size
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    LAMBDA = 100
    checkpoint_dir = 'training_checkpoints'
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

    generator = Generator(output_channels=3)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4,
                                                   beta_1=0.5)
    discriminator = Discriminator()
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4,
                                                       beta_1=0.5)
    # checkpoint = tf.train.Checkpoint(
    #     generator_optimizer=generator_optimizer,
    #     discriminator_optimizer=discriminator_optimizer,
    #     generator=generator,
    #     discriminator=discriminator)

    for epoch in range(epochs):
        start = time.time()

        # Train
        print(f"EPOCH {epoch}")
        for n, (input_image, target) in train_dataset.enumerate():
            print(f"Image {n}")
            if n == 5:
                break
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

        # Test on the same image so that the progress of the model can be
        # saving (checkpoint) the model every 20 epochs
        # if (epoch + 1) % 20 == 0:
        #     checkpoint.write(file_prefix=checkpoint_path)

        print(
            'Time taken for epoch {} is {} sec\n'.format(
                epoch + 1,
                time.time()-start))
    print("DONE")
