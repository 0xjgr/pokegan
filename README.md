# PokeGAN

*[Versión en español](README.es.md)*

A DCGAN trained from scratch for 10,000 epochs to generate 64x64 Pokémon
sprites that don't exist. It started as the final project for a deep learning
course and ended up being mostly an exercise in wrestling with GAN
instability, which is where the real learning happens.

![Epoch 5000](Epochs/epoch_05000.png)

## How it works

A GAN is two networks playing a zero-sum game:

- **The generator** starts from a 100-dimensional noise vector and, through
  transposed convolutions (`Conv2DTranspose`), projects it up to a 64x64x3
  image.
- **The discriminator** is a binary classifier trying to tell the real
  Pokémon in the dataset apart from the generator's fakes.

At each step the discriminator is trained to get it right and the generator
to make it fail. The balance between the two is the whole problem.

**Dataset:** about 800 images from the [Pokemon Images Dataset](https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset)
on Kaggle, resized to 64x64 and normalized to the [-1, 1] range.

## What broke along the way

This is the interesting part of the project. The three things that were hard:

### Mode collapse

In the early versions the generator found a greenish-red blob that fooled the
discriminator, and from then on it stopped exploring: it produced the same
image over and over. The model had found a comfortable minimum and stayed
there.

The fix was giving the generator an edge. I lowered the discriminator's
*learning rate* to 2e-5 versus the generator's 1e-4, which is the idea behind
**TTUR** (Two Time-Scale Update Rule), and raised the discriminator's
`Dropout` to 0.3. With the discriminator learning more slowly, the generator
has room to try out shapes before getting penalized.

### Checkerboard artifacts

Grid patterns showed up in the images, the classic `Conv2DTranspose` problem
when the kernel size isn't a multiple of the stride and the overlap ends up
uneven. It was fixed by adjusting kernels and strides, and by training longer
so `BatchNormalization` could stabilize the activations.

### Stability at 10,000 epochs

With perfect binary labels (0 and 1) the gradient becomes too steep and
training goes off the rails after a few thousand iterations. With **label
smoothing**, real images are labeled 0.9 instead of 1.0, and that's enough
for training to hold up for the full 10,000 epochs.

## Training progress

| Epoch 20 | Epoch 100 | Epoch 500 |
| :---: | :---: | :---: |
| ![](Epochs/epoch_00020.png) | ![](Epochs/epoch_00100.png) | ![](Epochs/epoch_00500.png) |

| Epoch 1000 | Epoch 3000 | Epoch 5000 |
| :---: | :---: | :---: |
| ![](Epochs/epoch_01000.png) | ![](Epochs/epoch_03000.png) | ![](Epochs/epoch_05000.png) |

All the saved checkpoints are in `Epochs/`, and
[`video/evolucion_pokemon.mp4`](video/evolucion_pokemon.mp4) has the full
evolution edited into a video.

## Generating images

The trained model is in `modelo/`. To get a grid of samples:

```bash
pip install -r requirements.txt
python generar.py --n 16 --output samples.png
```

Or from Python, if you'd rather use it directly:

```python
import tensorflow as tf

generator = tf.keras.models.load_model(
    "modelo/generador_pokemon_10000_final.h5", compile=False
)
noise = tf.random.normal([1, 100])
image = (generator(noise, training=False)[0] + 1) / 2   # from [-1,1] to [0,1]
```

`compile=False` stops Keras from trying to rebuild the optimizer, which isn't
needed for generating. The file is in Keras's legacy `.h5` format.

## Retraining

The full training is in `entrenamiento_dcgan.ipynb` (the notebook's comments
are in Spanish). You need to download the dataset from Kaggle and put it at
the path `DATA_PATH` points to. It's 10,000 epochs: on a decent GPU it's a
matter of hours; on CPU, don't even try.

## Architecture

| | Generator | Discriminator |
| --- | --- | --- |
| Input | 100-dim noise | 64x64x3 image |
| Blocks | Dense 8x8x256, 3x `Conv2DTranspose` (128, 64, 3) | 2x `Conv2D` (64, 128) |
| Activation | ReLU, `tanh` at the output | LeakyReLU(0.2) |
| Regularization | BatchNormalization | Dropout(0.3) |
| Optimizer | Adam, lr 1e-4, beta_1 0.5 | Adam, lr 2e-5, beta_1 0.5 |

Loss: binary cross-entropy from logits, with 0.9 label smoothing on real
samples.

## What worked and what didn't

What works: the model escapes mode collapse and generates variety, and it
captures the franchise's color palettes and organic silhouettes well. At 64x64
and at a glance, many samples pass for Pokémon.

What doesn't: at that resolution there's no real detail, and no creature has
coherent anatomy if you look closely. Going up to 128x128 needed deeper
filters and much more training than I had time for. I also didn't measure
anything objectively: there's no FID or Inception Score; evaluation was
looking at sample grids, which is fine for a class project but isn't
evaluation.

## Structure

```
entrenamiento_dcgan.ipynb   the full training, cell by cell
generar.py                  generates samples from the trained model
modelo/                     trained generator, 10,000 epochs
Epochs/                     samples saved during training
video/                      the evolution edited into a video
```

## References

- Ian J. Goodfellow et al. (2014), [*Generative Adversarial Nets*](https://arxiv.org/abs/1406.2661).
- Alec Radford et al. (2015), [*Unsupervised Representation Learning with Deep
  Convolutional Generative Adversarial Networks*](https://arxiv.org/abs/1511.06434),
  the DCGAN paper.
- Heusel et al. (2017), [*GANs Trained by a Two Time-Scale Update Rule*](https://arxiv.org/abs/1706.08500),
  where TTUR comes from.
- The [TensorFlow DCGAN tutorial](https://www.tensorflow.org/tutorials/generative/dcgan),
  which was the starting point.
- Dataset: [Pokemon Images Dataset](https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset), on Kaggle.

## License

The code is under the MIT license, copyright 0xjgr. See
[LICENSE](LICENSE).

That covers the code and nothing else. The trained model (`modelo/`) and the
generated images (`Epochs/`, `video/`) come from a dataset of Pokémon
artwork, which is owned by Nintendo, Game Freak and Creatures. I'm publishing
them as the outcome of an academic project, not as something you can freely
reuse. If you're going to do anything with them other than look at them, do
your own research.
