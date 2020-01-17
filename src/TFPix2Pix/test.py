from argparse import _SubParsersAction
from pathlib import Path

import tensorflow as tf
import logging
import signal
import sys

from .network.helpers import load_image
from .network.models import Generator
from .file_manager import save_pyplot


def args(sub_parser: _SubParsersAction) -> None:
    sub_parser.add_argument('--weights', type=str,
                            dest='weights',
                            required=True,
                            help='Required. Weights file path')
    sub_parser.add_argument('--input', type=str,
                            dest='input',
                            required=True,
                            help='Required. Input images dir path')
    sub_parser.add_argument('--output', type=str,
                            dest='output',
                            required=True,
                            help='Required. Output images dir path')
    sub_parser.add_argument('--batch-size', type=int,
                            dest='batch_size',
                            default=1,
                            help='Default = 1. Batch Size for Testing.')
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


# @tf.function
def infer(checkpoint: Path,
          input_path: Path,
          output_path: Path,
          batch_size: int,
          gpu: bool,) -> bool:
    """
    @param checkpoint: can either be a specific path to a checkpoint file
                       or a dir containing multiple. In the later case,
                       the latest checkpoint in that dir is chosen.
    @param input_path: children should all be images
    """
    signal.signal(signal.SIGINT, control_c_handler)
    print("\n---------------------------------")
    print("TFPix2Pix Testing")
    print("---------------------------------\n")
    logging.debug("TFPix2Pix Test: infer arguments: \n"
                  f"@param checkpoint   |  {checkpoint} \n"
                  f"@param input_path   |  {input_path} \n"
                  f"@param output_path  |  {output_path} \n"
                  f"@param batch_size   |  {batch_size} \n"
                  f"@param gpu          |  {gpu} \n")

    if checkpoint.is_dir():
        checkpoint = tf.train.latest_checkpoint(str(checkpoint))
    elif checkpoint.suffix != '.ckpt':
        logging.critical(f"TFPix2Pix Testing: checkpoint path {checkpoint} " +
                         "is invalid")
        return False

    dataset = tf.data.Dataset.list_files(
        str(input_path / '*'))
    dataset = dataset.map(load_image)
    dataset = dataset.batch(batch_size)
    logging.debug("TFPix2Pix Test:  Dataset created.\n\n")

    device = '/device:CPU:0' if not gpu else '/device:GPU:0'
    logging.debug("TFPix2Pix Test: Starting try block")
    try:
        with tf.device(device):
            model = Generator(output_channels=3)
            logging.debug("TFPix2Pix Test: Generator Created")
            model.load_weights(checkpoint)
            logging.info("TFPix2Pix Test: weights loaded.")
            for n, (image, name) in dataset.enumerate():
                prediction = model(image, training=False)
                if isinstance(prediction, list):
                    if len(prediction) > 1:
                        prediction = prediction[0]
                    else:
                        logging.critical("TFPix2Pix Test: prediction " +
                                         "returned was a list of size 0")
                    return False
                path = output_path / 'test.jpg'
                save_pyplot(file_name=str(path),
                            image=prediction)
                logging.info(f"TFPix2Pix Test: image saved to {path}")
    except Exception as e:
        logging.error(f"TFPix2Pix Test: Exception: {e}")
        return False
    return True
