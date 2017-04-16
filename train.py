from chainer.datasets import get_cifar10
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions

import config
from lib.datasets import get_celeba
from lib.extensions import GeneratorSampler
from lib.iterators import RandomNoiseIterator
from lib.iterators import UniformNoiseGenerator
from lib.models import Discriminator
from lib.models import Generator
from lib.updater import BEGANUpdater


if __name__ == '__main__':
    args = config.parse_args()

    if args.dataset == 'celeba':
        train = get_celeba(args.celeba_root, args.celeba_scale, crop='face')
    elif args.dataset == 'cifar10':
        train, _ = get_cifar10(withlabel=False, scale=2.0)
        train -= 1.0

    train_iter = iterators.SerialIterator(train, args.batch_size)
    z_iter = RandomNoiseIterator(UniformNoiseGenerator(-1, 1, args.n_z),
                                 args.batch_size)

    g = Generator(
            n=args.g_n,
            out_size=train[0].shape[1],
            out_channels=train[0].shape[0],
            block_size=args.g_block_size,
            embed_size=args.g_embed_size)

    d = Discriminator(
            n=args.d_n,
            h=args.n_h,
            in_size=train[0].shape[1],
            in_channels=train[0].shape[0],
            block_size=args.d_block_size,
            embed_size=args.d_embed_size)

    optimizer_generator = optimizers.Adam(alpha=args.g_lr, beta1=0.5)
    optimizer_discriminator = optimizers.Adam(alpha=args.d_lr, beta1=0.5)
    optimizer_generator.setup(g)
    optimizer_discriminator.setup(d)

    updater = BEGANUpdater(
        iterator=train_iter,
        noise_iterator=z_iter,
        optimizer_generator=optimizer_generator,
        optimizer_discriminator=optimizer_discriminator,
        generator_lr_decay_interval=args.g_lr_decay_interval,
        discriminator_lr_decay_interval=args.g_lr_decay_interval,
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
