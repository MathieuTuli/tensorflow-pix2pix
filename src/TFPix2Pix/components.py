from enum import Enum

import logging


class ImageDirection(Enum):
    AtoB = 0
    BtoA = 1

    def __str__(self):
        return self.name


class LogLevel(Enum):
    '''
    What the stdlib did not provide!
    '''
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    def __str__(self):
        return self.name
