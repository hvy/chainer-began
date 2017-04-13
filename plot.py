import math

import numpy
from PIL import Image


def save_ims(filename, ims):
    n, c, w, h = ims.shape

    rows = math.ceil(math.sqrt(n))
    cols = rows if n % rows == 0 else rows - 1

    if c == 3:
        ims = ims.reshape((rows, cols, 3, h, w))
        ims = ims.transpose(0, 3, 1, 4, 2)
        ims = ims.reshape((rows * h, cols * w, 3))
    else:
        ims = ims.reshape((rows, cols, 1, h, w))
        ims = ims.transpose(0, 3, 1, 4, 2)
        ims = ims.reshape((rows * h, cols * w))

    ims = ims.astype(numpy.uint8)

    Image.fromarray(ims).save(filename)
