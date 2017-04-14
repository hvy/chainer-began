from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions

from lib import datasets
from lib.extensions import GeneratorSampler
from lib.iterators import RandomNoiseIterator
from lib.iterators import UniformNoiseGenerator
from lib.models import Discriminator
from lib.models import Generator
from lib.updater import BEGANUpdater
import train_config


if __name__ == '__main__':
    args = train_config.parse_args()

    train = datasets.get_celeba(args.celeba_root, args.celeba_scale,
                                crop='face')

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

    trainer = training.Trainer(updater, out=args.out_dir,
                               stop_trigger=(args.iterations, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(1000, 'iteration')))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PrintReport(['epoch',
                                           'iteration',
                                           'convergence',
                                           'gen/loss',
                                           'dis/loss',
                                           'k']))
    trainer.extend(GeneratorSampler(), trigger=(1000, 'iteration'))
    trainer.run()
