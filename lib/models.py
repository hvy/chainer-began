from chainer import Chain
from chainer import functions as F
from chainer import links as L


class Generator(Chain):
    def __init__(self, n):
        super().__init__(decoder=Decoder(n))

    def __call__(self, z):
        return self.decoder(z)


class Discriminator(Chain):
    def __init__(self, *, n, h):
        super().__init__(encoder=Encoder(n, h), decoder=Decoder(n))

    def __call__(self, x):
        return self.decoder(self.encoder(x))


class Decoder(Chain):
    def __init__(self, n):
        super().__init__(
            fc=L.Linear(None, 8*8*n),
            c1=L.Convolution2D(n, n, 3, stride=1, pad=1),
            c2=L.Convolution2D(n, n, 3, stride=1, pad=1),
            c3=L.Convolution2D(n, n, 3, stride=1, pad=1),
            c4=L.Convolution2D(n, n, 3, stride=1, pad=1),
            c5=L.Convolution2D(n, n, 3, stride=1, pad=1),
            c6=L.Convolution2D(n, n, 3, stride=1, pad=1),
            c7=L.Convolution2D(n, n, 3, stride=1, pad=1),
            c8=L.Convolution2D(n, n, 3, stride=1, pad=1),
            c9=L.Convolution2D(n, 3, 3, stride=1, pad=1))

        self.n = n

    def __call__(self, x):
        h = F.reshape(self.fc(x), (x.shape[0], self.n, 8, 8))

        h = F.elu(self.c1(h))
        h = F.elu(self.c2(h))
        h = F.unpooling_2d(h, ksize=2, stride=2, cover_all=False)

        h = F.elu(self.c3(h))
        h = F.elu(self.c4(h))
        h = F.unpooling_2d(h, ksize=2, stride=2, cover_all=False)

        h = F.elu(self.c5(h))
        h = F.elu(self.c6(h))
        h = F.unpooling_2d(h, ksize=2, stride=2, cover_all=False)

        h = F.elu(self.c7(h))
        h = F.elu(self.c8(h))
        h = self.c9(h)
        # assert(h.shape[1:] == (3, 64, 64))

        return h


class Encoder(Chain):
    def __init__(self, n, h):
        super().__init__(
            c0=L.Convolution2D(3, n, 3, stride=1, pad=1),
            c1=L.Convolution2D(n, n, 3, stride=1, pad=1),
            c2=L.Convolution2D(n, 2*n, 3, stride=1, pad=1),
            c3=L.Convolution2D(2*n, 2*n, 3, stride=1, pad=1),
            c4=L.Convolution2D(2*n, 3*n, 3, stride=1, pad=1),
            c5=L.Convolution2D(3*n, 3*n, 3, stride=1, pad=1),
            c6=L.Convolution2D(3*n, 4*n, 3, stride=1, pad=1),
            c7=L.Convolution2D(4*n, 4*n, 3, stride=1, pad=1),
            c8=L.Convolution2D(4*n, 4*n, 3, stride=1, pad=1),
            fc=L.Linear(8*8*4*n, h))

        self.n = n
        self.h = h

    def __call__(self, x):
        h = F.elu(self.c0(x))

        h = F.elu(self.c1(h))
        h = F.elu(self.c2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.elu(self.c3(h))
        h = F.elu(self.c4(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.elu(self.c5(h))
        h = F.elu(self.c6(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.elu(self.c7(h))
        h = F.elu(self.c8(h))
        h = self.fc(h)

        return h
