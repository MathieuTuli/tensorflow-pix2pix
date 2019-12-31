from typing import NamedTuple

from enum import Enum


class ImageDirection(Enum):
    AtoB = 0
    BtoA = 1

    def __str__(self):
        return self.name
