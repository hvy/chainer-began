import chainer
import numpy as np
from PIL import Image


class ImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, paths, size):
        self.paths = paths
        self.size = size

    def __len__(self):
        return len(self.paths)

    def get_example(self, i):
        path = self.paths[i]
        im_file = Image.open(path)
        im_file = im_file.resize(self.size, Image.ANTIALIAS)

        try:
            im = np.asarray(im_file, dtype=np.float32)
        finally:
            if hasattr(im_file, 'close'):
                im_file.close()

        # NOTE: Uncomment if the image is modified
        # im = im.copy()

        im = im.transpose((2, 0, 1))

        # Rescale to [-1, 1]
        im *= (2 / 255)
        im -= 1

        return im
