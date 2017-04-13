import glob

from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions

import datasets
from extensions import GeneratorSample
from iterators import RandomNoiseIterator
from iterators import UniformNoiseGenerator
from models import Discriminator
from models import Generator
import train_config
from updater import BEGANUpdater


def get_celeba(root, size=(64, 64)):
    paths = glob.glob('{}/Img/img_align_celeba_png/*.png'
                      .format(root))
    return datasets.ImageDataset(paths, size)


if __name__ == '__main__':
    args = train_config.parse_args()

    train = get_celeba(args.celeba_root)
    train_iter = iterators.SerialIterator(train, args.batch_size)
    z_iter = RandomNoiseIterator(UniformNoiseGenerator(-1, 1, args.n_z),
                                 args.batch_size)

    optimizer_generator = optimizers.Adam(alpha=args.g_lr, beta1=0.5)
    optimizer_discriminator = optimizers.Adam(alpha=args.d_lr, beta1=0.5)
    optimizer_generator.setup(Generator(n=args.g_n))
    optimizer_discriminator.setup(Discriminator(n=args.d_n, h=args.n_h))

    updater = BEGANUpdater(
        iterator=train_iter,
        noise_iterator=z_iter,
        optimizer_generator=optimizer_generator,
        optimizer_discriminator=optimizer_discriminator,
        gamma=args.gamma,
        k_0=args.k_0,
        lambda_k=args.lambda_k,
        loss_norm=args.loss_norm,
        device=args.gpu)

    trainer = training.Trainer(updater, stop_trigger=(args.epochs, 'epoch'))

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PrintReport(['iteration',
                                           'gen/loss',
                                           'dis/loss',
                                           'k']))
    trainer.extend(GeneratorSample(), trigger=(100, 'iteration'))

    trainer.run()
