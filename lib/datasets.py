import glob

import chainer
import numpy as np
from PIL import Image


class ImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, paths, size=None, crop=None):
        self.paths = paths
        self.size = size
        self.crop = crop

    def __len__(self):
        return len(self.paths)

    def get_example(self, i):
        path = self.paths[i]

        with Image.open(path) as f:
            if self.crop is not None:
                f = f.crop(self.crop)
            if self.size is not None:
                f = f.resize(self.size, Image.ANTIALIAS)
            im = np.asarray(f, dtype=np.float32)

        # NOTE: Uncomment if it is necessary to modify the image
        # im = im.copy()

        im = im.transpose((2, 0, 1))

        # Rescale from [0, 255] to [-1, 1]
        im *= (2 / 255)
        im -= 1

        return im


def get_celeba(root, size=(64, 64), crop=(25, 50, 25+128, 50+128)):
    paths = glob.glob('{}/Img/img_align_celeba_png/*.png' .format(root))
    return ImageDataset(paths, size, crop)
