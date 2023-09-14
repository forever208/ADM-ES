# Download datasets


## CIFAR-10

For CIFAR-10, we created a script [cifar10.py](cifar10.py)` that creates `cifar_train` and `cifar_test` directories. These directories contain files named like `truck_49997.png`, so that the class name is discernable to the data loader.

The `cifar_train` and `cifar_test` directories can be passed directly to the training scripts via the `--data_dir` argument.

[cifar10.py](cifar10.py) also support transforming the dataset to npz file.

## ImageNet 32x32
First download ImageNet-32 from [ImageNet official website](https://image-net.org/download.php), then use our script [imagenet32_npz.py](Imagenet32_npz.py) to pack all images into a .npz file


## LSUN tower 64x64

To download and pre-process LSUN tower, clone [fyu/lsun](https://github.com/fyu/lsun) on GitHub and run their download script `python download.py -c tower`. The result will be an "lmdb" database zip file, unzip it to get a `tower_train_lmdb` folder. You can pass this to our [lsun_tower.py](lsun_tower.py) script to the folder `lsun_tower_train` that contains all jpg format images:

```
python lsun_tower.py --image-size 64 --prefix tower tower_train_lmdb lsun_tower_train
```


## CelebA 64x64
First download the raw CelebA from [GoogleDrive](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ), then use our [celeba64_npz.py](celeba64_npz.py) script to do the preprocessing.

Alternatively, using the scrips of [DDIM](https://github.com/ermongroup/ddim/blob/main/datasets/celeba.py) to download and preprocess CelebA 64 is also recommended.


## CelebA-HQ256
You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256) and process the images into npz file using script [celebaHQ256_npz.py](celebaHQ256_npz.py)

## FFHQ128, FFHQ256
Kaggle provides the direct download for [FFHQ128](https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set) [FFHQ256](https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only)
See the scripts [ffhq128_npz.py](./ffhq128_npz.py) and [ffhq256_npz.py](./ffhq256_npz.py)

## FFHQ 64X64, AFHQv2
Additionally, I found that [EDM repo](https://github.com/NVlabs/edm) provided some useful tools to prepare the datasets, please feel free to take a look
