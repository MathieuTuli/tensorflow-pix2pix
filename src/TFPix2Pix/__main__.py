from argparse import ArgumentParser
from pathlib import Path

import logging
import signal
import sys

from .components import LogLevel
from .helpers import file_exists
from .train import args as train_args, fit
from .test import args as test_args

print("\n---------------------------------")
print("TFPix2Pix")
print("---------------------------------\n")

parser = ArgumentParser(description=__doc__)
parser.add_argument(
    '-vv', '--very-verbose', action='store_true',
    dest='very_verbose',
    help="Set verbose. In effect, set --log-level to DEBUG.")
parser.add_argument(
    '-v', '--verbose', action='store_true',
    dest='verbose',
    help="Set verbose. In effect, set --log-level to INFO.")
parser.set_defaults(verbose=False)
parser.set_defaults(very_verbose=False)
parser.add_argument('--log-level', type=LogLevel.__getitem__,
                    default=LogLevel.INFO,
                    choices=LogLevel.__members__.values(),
                    dest='log_level',
                    help="Log level.")
subparser = parser.add_subparsers(dest='command')
train_subparser = subparser.add_parser(
    'train', help='Train commands')
train_args(train_subparser)
test_subparser = subparser.add_parser(
    'test', help='Test commands')
test_args(test_subparser)

args = parser.parse_args()
if args.log_level == LogLevel.DEBUG or args.very_verbose:
    logging.root.setLevel(logging.DEBUG)
elif args.log_level == LogLevel.INFO or args.verbose:
    logging.root.setLevel(logging.INFO)
elif args.log_level == LogLevel.WARNING:
    logging.root.setLevel(logging.WARNING)
elif args.log_level == LogLevel.ERROR:
    logging.root.setLevel(logging.ERROR)
elif args.log_level == LogLevel.CRITICAL:
    logging.root.setLevel(logging.CRITICAL)
else:
    logging.root.setLevel(logging.INFO)
    logging.warning(
        f"Pix2PIx: Log level \"{args.log_level}\" unknown, defaulting" +
        " to INFO.")
logging.info(
    f"TFPix2Pix: Log level set to \"{logging.getLogger()}\"")


def control_c_handler(_signal, frame):
    print("\n---------------------------------")
    print("TFPix2Pix: Ctrl-C. Shutting Down.")
    print("---------------------------------")
    sys.exit(0)


signal.signal(signal.SIGINT, control_c_handler)


if __name__ == "__main__":
    command = '' if args.command is None else str(args.command).lower()
    if command == 'train':
        data = Path(args.data)
        checkpoint = Path(args.checkpoint)

        if file_exists(data):
            if file_exists(checkpoint):
                fit(dataset_path=str(data),
                    checkpoint_path=str(checkpoint),
                    image_direction=args.image_direction,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    buffer_size=args.buffer_size,
                    _lambda=args._lambda,
                    checkpoint_save_freq=args.save_freq,
                    gpu=args.gpu,
                    output_channels=args.output_channels)
            else:
                logging.critical(
                        f"TFPix2Pix: Checkpoint path {checkpoint} doesn't " +
                        "exist")
                sys.exit(0)
        else:
            logging.critical(f"TFPix2Pix: Data path {data} doesn't exist")
            sys.exit(0)
    elif command == 'test':
        ...
    else:
        logging.critical(
            f"TFPix2Pix: Command \"{command}\" unknown. " +
            "'train' and 'test' allowed.")
        sys.exit(0)
