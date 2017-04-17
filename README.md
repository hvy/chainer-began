# BEGAN: Boundary Equilibrium Generative Adversarial Networks

Chainer implementation of the [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/abs/1703.10717) by David Berthelot et al. Note that this is not the official implementation.

## Results

Training results using the default parameters in [config.py](config.py).

![](images/sample_1142000.png)

Samples generated from unif(-1, 1) noise vector after around 1e6 iterations.

![](images/loss.png)

The generator and discriminator losses as well as the global loss based on the process error. The visual fidelity of the generated samples keep improving throughout the training.

The Adam learning rate is kept constant at 5e-5, in contrast to the paper. Starting with a higher learning rate and then decaying it is expected to improve the results and converge faster.

## Train

### CelebA

To train the model with CelebA as in the results above, download the aligned and cropped version of [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and unarchive the whole dataset.

Images will during training be **cropped** to remove some of the background and then **rescaled** to e.g. (64, 64) or (128, 128) pixels. A larger scale will automatically result in a deeper architecture.

```bash
python train.py --celeba-root celeba/CelebA --celeba-scale 64 --batch-size 16 --iterations 10000 --gpu 1
```

### CIFAR-10

To train the model with [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) with (32, 32) images, simply run the following command. Chainer will download and cache the dataset for you.

Note that with the same default hyperparameters as for CelebA, the results look fairly poor with centered blob-like objects.

```bash
python train.py --dataset cifar10 --batch-size 16 --iterations 10000 --gpu 1
```

Images are randomly sampled from the generator every certain number of iterations, and saved under a subdirectory `result`.
