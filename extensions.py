import os

from chainer import training
from chainer import cuda
from chainer.training import extension
import numpy

import plot


class GeneratorSample(extension.Extension):
    def __init__(self, dirname='sample', sample_format='png'):
        self._dirname = dirname
        self._sample_format = sample_format

    def __call__(self, trainer):
        dirname = os.path.join(trainer.out, self._dirname)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        x = self.sample(trainer)

        # Assume that x is somewhere in the range [-1, 1] and rescale to
        # [0, 255]
        x = numpy.clip((x + 1.0) / 2.0 * 255.0, 0.0, 255.0)

        filename = 'sample_{}.{}'.format(trainer.updater.iteration,
                                  self._sample_format)
        filename = os.path.join(dirname, filename)
        plot.save_ims(filename, x)

    def sample(self, trainer):
        x = trainer.updater.sample()
        x = x.data
        if cuda.available and isinstance(x, cuda.ndarray):
            x = cuda.to_cpu(x)
        return x
