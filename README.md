# BEGAN: Boundary Equilibrium Generative Adversarial Networks

Chainer implementation of the [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/abs/1703.10717) by David Berthelot et al. Note that this is not the official implementation.

## Run

The model is trained with the aligned and cropped version of [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Download and unarchive the whole dataset first.

Images will during training be center cropped to remove some of the background and then rescaled to (64, 64). Randomly images are sampled from the generator every certain number of iterations, and saved under a subdirectory `result/` (which is created automatically).

```bash
python train.py --celeba-root celeba/CelebA --batch-size 16 --iterations 10000 --gpu 1
```

## Training

![](images/sample_1142000.png)

Images sampled from the generator with random noise from unif(-1, 1) after 1142000 iterations using the default parameters in [config.py](config.py).

![](images/loss.png)

Plotting the losses for the generator and the discriminator as well as the global loss based on the process error. The visual fidelity of the generated samples keep improving even though the curves seem to converge.

The Adam learning rates are kept constant 5e-5, in contrast to the paper. Starting with a higher learning rate and then decaying it is expected to improve the results and converge faster.
