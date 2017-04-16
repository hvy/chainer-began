from math import log2

from chainer import Chain
from chainer import functions as F
from chainer import links as L


class Generator(Chain):
    def __init__(self, *, n, out_size, out_channels, block_size=2,
                 embed_size=8):
        super().__init__(decoder=Decoder(
                n=n,
                out_size=out_size,
                out_channels=out_channels,
                embed_size=embed_size,
                block_size=block_size))

    def __call__(self, z):
        return self.decoder(z)


class Discriminator(Chain):
    def __init__(self, *, n, h, in_size, in_channels, block_size=2,
                 embed_size=8):
        super().__init__(encoder=Encoder(
                n=n,
                h=h,
                in_size=in_size,
                in_channels=in_channels,
                embed_size=embed_size,
                block_size=block_size
            ), decoder=Decoder(
                n=n,
                out_size=in_size,
                out_channels=in_channels,
                embed_size=embed_size,
                block_size=block_size))

    def __call__(self, x):
        return self.decoder(self.encoder(x))


class Encoder(Chain):
    def __init__(self, n, h, in_size, in_channels, embed_size, block_size):
        super().__init__(
            l0=L.Convolution2D(in_channels, n, 3, stride=1, pad=1),
            ln=L.Linear(None, h))

        self.n_blocks = int(log2(in_size / embed_size)) + 1
        self.block_size = block_size

        for i in range(self.n_blocks):
            n_in = (i + 1) * n
            n_out = (i + 2) * n if i < self.n_blocks - 1 else n_in
            for j in range(block_size - 1):
                self.add_link('c{}'.format(i * block_size + j),
                              L.Convolution2D(n_in, n_in, 3, stride=1, pad=1))
            self.add_link('c{}'.format(i * block_size + block_size - 1),
                          L.Convolution2D(n_in, n_out, 3, stride=1, pad=1))

    def __call__(self, x):
        h = F.elu(self.l0(x))

        for i in range(self.n_blocks):
            for j in range(self.block_size):
                h = getattr(self, 'c{}'.format(i * self.block_size + j))(h)
                h = F.elu(h)
            if i < self.n_blocks - 1:
                h = F.max_pooling_2d(h, ksize=2, stride=2)

        return self.ln(h)


class Decoder(Chain):
    def __init__(self, n, out_size, out_channels, embed_size, block_size):
        super().__init__(
                l0=L.Linear(None, n * embed_size * embed_size),
                ln=L.Convolution2D(n, out_channels, 3, stride=1, pad=1))

        self.embed_shape = (n, embed_size, embed_size)
        self.n_blocks = int(log2(out_size / embed_size)) + 1
        self.block_size = block_size

        for i in range(self.n_blocks * block_size):
            self.add_link('c{}'.format(i),
                          L.Convolution2D(n, n, 3, stride=1, pad=1))

    def __call__(self, x):
        h = F.reshape(self.l0(x), ((x.shape[0],) + self.embed_shape))

        for i in range(self.n_blocks):
            for j in range(self.block_size):
                h = F.elu(getattr(self, 'c{}'.format(i*j+j))(h))
            if i < self.n_blocks - 1:
                h = F.unpooling_2d(h, ksize=2, stride=2, cover_all=False)

        return self.ln(h)
