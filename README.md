# alexnet_pytorch
Implementation of `ImageNet Classification with Deep Convolutional Neural Networks` by `Alex Krizhevsky`, `Ilya Sutskever` and `Geoffrey E. Hinton`
in a plain `PyTorch`.

# Results
To reproduce results i used only information from the paper.

## Dataset
I used the following [dataset](https://huggingface.co/datasets/ILSVRC/imagenet-1k) from HuggingFace.

For training i preprocessed images offline using `preprocess_imagenet.py` script. The script performs:
1) Rescale images such that the shortest side of the image is equal to 256
   and other side is being kept so the aspect ratio isn't changed;
2) Center crop image from step 1.

This saves about 2 minutes per epoch.

## Metrics
|                          | Val error rate @ 1 (10-crop) | Val error rate @ 5 (10-crop) |
|--------------------------|------------------------------|------------------------------|
| AlexNet (mine) 90 epoch  | 0.467                        | 0.2269                       |
| AlexNet (paper) 90 epoch | 0.407                        | 0.182                        |
| AlexNet (mine) 120 epoch | 0.417                        | 0.193                        |

As you can see i was not able to reproduce the results using modern PyTorch with training during 90 epoch. But if i train
120 epoch i get almost target paper metric.

## Discrepancy from the article
I tried to strictly follow the article but in some cases i just can not do this. Here are the stuff which are not the same
as in paper:
1. Weights initialization. I implemented paper weights initialization, but the model simply doesn't train with it. So i
used standard `PyTorch` weights initialization. You can check it no `source/models/alexnet.py`;
2. `PCA` data augmentation. I used `FancyPCA` from awesome [`albumentations`](https://github.com/albumentations-team/albumentations) library, but it's implemented in such a way that principal components have been found for the each single image and used them in actual data augmentation process. However in paper authors precomputes eigen vectors and eigen values across all training set. You can find these precomputed vectors and values [here](https://github.com/facebookarchive/fb.resnet.torch/blob/985e569d468d23baeef5b952954dcdf3c61c5e73/datasets/imagenet.lua#L71).
3. Learning rate scheduler. Authors mentioned that they schedule learning rate manually if validation error rate is not improving. But i used `torch.optim.lr_scheduler.ReduceLROnPlateau` with default arguments, so there may be some descrepancy.
4. The number of epoch. From the previous section you can see that to reach paper metrics it's needed to train around 120 epoch, not 90.

So, i tried to use same hyperparameters, architecture and other stuff as much as i can.

# How to use

## Installation
You need `Python 3.12`, `CUDA 12.4` and corresponding `virtualenv` module.
Requirements listed in `requirements.txt` file and dev-requirements are listed in `requirements.dev.txt`. Simply do `pip install -r requirements.dev.txt` after creating the fresh env and it's all should be ok.

I used Windows 11, so i don't know will it work on Linux.

## Training
Go to the `config.yml` file and fill it according to you desires. Then simply run `python train.py`.
`train.py` script also accepts optional parameter - the path to the config file:
```
python train.py -c path\to\config.yml
```

## Classifing the image
You can use `classify_image.py` to classify single image with trained model:
```
python classify_image.py \
 -i path\to\image.jpeg \
 -w path\to\trained_weight.pt \
 -d cpu \ # or cuda:0
 -k 5
```
the script output classes to the console.