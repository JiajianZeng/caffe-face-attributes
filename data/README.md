This directory holds (*after you download them*):
- Caffe models pre-trained on ImageNet
- Faster R-CNN models
- Symlinks to datasets

To download Caffe models (ZF, VGG16) pre-trained on ImageNet, run:

```
./data/scripts/fetch_imagenet_models.sh
```

This script will populate `data/imagenet_models`.

To download Faster R-CNN models trained on VOC 2007, run:

```
./data/scripts/fetch_faster_rcnn_models.sh
```

This script will populate `data/faster_rcnn_models`.

In order to train and test with CelebAdevkit, you will need to establish symlinks.
From the `data` directory (`cd data`):

```
ln -s /your/path/to/Attributedevkit CelebAdevkit
```
