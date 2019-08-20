# Mention all the code that you want to export to the users
from .build_dataset import (
    gather_data, split_and_store
)
import re
# -----------------  Define your parser here ----------------- #

# parser = lambda x: x

def parser(x: str, image_type: str) -> str:
    """
    Converts the string name into the required form
    :param x: file name
    :param image_type: Extension of the image
    :return: processed file name
    """
    regex = rf"([0-9])_IMG_(\d+).*\.({image_type})"
    matches = re.match(regex, x)
    return "{}_{}.{}".format(*matches.groups())

#--------------------------------------------------------------#

__all__ = [
    'gather_data',
    'split_and_store',
    'parser'
]

