## Evaluations

To compare different generative models, we use FID, sFID, Precision, Recall, and Inception Score. These metrics can all be calculated using batches of samples, which we store in `.npz` (numpy) files.

## Installation
Install the dependencies by `pip install -r requirements.txt`

you might encounter the bug: Could not load dynamic library 'libcudart.so.11.0 etc.
<br>Instead of using `pip install -r requirements.txt`, try the installation by:
```
$ pip install tensorflow==2.4
$ conda install cudatoolkit=11.0
$ conda install -c conda-forge cudnn
```

(refer to [libcudart.so.11.0](https://github.com/tensorflow/tensorflow/issues/45930) and [libcudart.so.8](https://github.com/tensorflow/tensorflow/issues/45200) for details)


## Evaluations for CIFAR-10
We need 2 things to compute FID: reference_batch (training data npz file) and sample_batch (your generated image npz file).
<br> We use the whole training set as the reference batch, except for LSUN tower where we use 50k images as the reference.

First, generate the reference batch `npz` sample using the script from the folder `./datasets/`
<br>For example, you should pack the whole cifar-10 training dataset and save it into `cifar_train.npz`.

Then, run the `evaluator.py` to compute FID:

```shell
mpiexec -n 1 python evaluator.py \
 ../datasets/cifar10_train.npz ../sample_10steps/samples_50000x32x32x3.npz
```

The output of the script will look something like this:

```
warming up TensorFlow...
computing reference batch activations ../datasets/cifar10_train.npz...
computing/reading reference batch statistics...
computing sample batch activations ../sample_epsilon_17/samples_50000x32x32x3.npz...
computing/reading sample batch statistics...
Computing evaluations...
Inception Score: 9.697924613952637
FID: 2.174162986785973
sFID: 3.8872043342477127
Precision: 0.68422
Recall: 0.6126

timesteps: 100
```

## Evaluations for large datasets
Since computing the statistics for the reference batch is also time-consuming on large datasets,
we save the statistics at the first time of FID computing by setting `--read_ref_statis = False`.
<br>The code will automatically load the reference statistics for the rest of FID computation by setting `--read_ref_statis = Ture`.
<br> For example, on celeba256 dataset, run the script by:

```shell
mpiexec -n 1 python evaluator_celeba256.py \
REFERENCH_BATCH ../celeba256/50000x256x256x3-samples.npz --read_ref_statis True
```

## Evaluations for ImageNet
For ImageNet datasets, OpenAI provides the statistics of the reference batch, download their [statistical files](https://github.com/openai/guided-diffusion/tree/main/evaluations) and set as reference batch.

Then, run the FID computation by:
```shell
mpiexec -n 1 python evaluator.py \
./VIRTUAL_imagenet64_labeled.npz ./imagenet64/samples_50000x64x64x3.npz
```


