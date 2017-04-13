from chainer import cuda
from chainer import functions as F
from chainer import reporter
from chainer import training
from chainer import Variable


def optimize(optimizer, loss):
    optimizer.target.cleargrads()
    loss.backward()
    optimizer.update()


class BEGANUpdater(training.StandardUpdater):
    def __init__(self, *, iterator, noise_iterator, optimizer_generator,
                 optimizer_discriminator, gamma, k_0, lambda_k, loss_norm,
                 device=-1):

        iterators = {'main': iterator, 'z': noise_iterator}
        optimizers = {'gen': optimizer_generator,
                      'dis': optimizer_discriminator}

        super().__init__(iterators, optimizers, device=device)

        self.k = k_0
        self.lambda_k = lambda_k
        self.gamma = gamma
        self.loss_norm = loss_norm

        if device >= 0:
            cuda.get_device(device).use()
            for optimizer in optimizers.values():
                optimizer.target.to_gpu()

    @property
    def generator(self):
        return self._optimizers['gen'].target

    @property
    def discriminator(self):
        return self._optimizers['dis'].target

    @property
    def optimizer_generator(self):
        return self._optimizers['gen']

    @property
    def optimizer_discriminator(self):
        return self._optimizers['dis']

    def z_batch(self):
        z = self._iterators['z'].next()
        return self._convert_batch(z)

    def x_batch(self):
        x = self._iterators['main'].next()
        return self._convert_batch(x)

    def _convert_batch(self, x):
        x = self.converter(x, self.device)
        return Variable(x)

    def pixel_wise_loss(self, x, y):
        if self.loss_norm == 1:
            return F.mean_absolute_error(x, y)
        elif self.loss_norm == 2:
            return F.mean_squared_error(x, y)
        else:
            raise ValueError('Invalid norm {}'.format(self.loss_norm))

    def update_core(self):
        loss_d, loss_real, _ = self.update_discriminator()
        loss_d, loss_real = loss_d.data, loss_real.data

        loss_g = self.update_generator()
        loss_g = loss_g.data

        self.k += self.lambda_k * ((self.gamma * loss_real) - loss_g)

        reporter.report({'loss': loss_d}, self.discriminator)
        reporter.report({'loss': loss_g}, self.generator)
        reporter.report({'k': self.k})

    def update_discriminator(self):
        # Reconstruction loss for real data
        x_real = self.x_batch()
        x_real_recon = self.discriminator(x_real)
        loss_real = self.pixel_wise_loss(x_real, x_real_recon)

        # Reconstruction loss for generated data
        z = self.z_batch()
        x_fake = self.generator(z)
        x_fake_recon = self.discriminator(x_fake)
        loss_fake = self.pixel_wise_loss(x_fake, x_fake_recon)

        # Total discriminator loss
        loss = loss_real - (self.k * loss_fake)

        optimize(self.optimizer_discriminator, loss)

        return loss, loss_real, loss_fake

    def update_generator(self):
        # Total generator loss
        z = self.z_batch()
        x_fake = self.generator(z)
        x_fake_recon = self.discriminator(x_fake)
        loss = self.pixel_wise_loss(x_fake, x_fake_recon)

        optimize(self.optimizer_generator, loss)

        return loss

    def sample(self):
        z = self.z_batch()
        return self.generator(z)
