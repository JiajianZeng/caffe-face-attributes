### Disclaimer

This implementation is based on [*Faster* R-CNN](https://github.com/rbgirshick/py-faster-rcnn). The details were described in my [rejected CVPR 2017 submission](https://drive.google.com/open?id=1L2I8Bt-ekJkW8PJzcPqPuQlbUB1Nazxk).

### Contents
1. [Requirements: software](#requirements-software)
2. [Basic installation](#installation-sufficient-for-the-demo)
3. [Training and testing](#beyond-the-demo-installation-for-training-and-testing-models)
4. [Usage](#usage)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```

  You can download [Makefile.config](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/Makefile.config) provided by the author of *Faster* R-CNN for reference.
  
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`

### Installation (sufficient for the demo)

1. Clone the caffe-face-attributes repository
  ```
  git clone https://github.com/JiajianZeng/caffe-face-attributes.git
  ```
  
2. We'll call the directory that you cloned caffe-face-attributes into `CFA_ROOT`

3. Build the Cython modules
  ```
  cd $CFA_ROOT/lib
  make
  ```
  
4. Build Caffe and pycaffe
  ```Shell
  cd $CFA_ROOT/caffe
  # Now follow the Caffe installation instructions here:
  # http://caffe.berkeleyvision.org/installation.html

  # If you're experienced with Caffe and have all of the requirements installed
  # and your Makefile.config in place, then simply do:
  make -j8 && make pycaffe
  ```

5. Download pre-computed Faster R-CNN detectors
  ```
  cd $CFA_ROOT
  ./data/scripts/fetch_faster_rcnn_models.sh
  ```
    
  This will populate the `$CFA_ROOT/data` folder with `faster_rcnn_models`. See `data/README.md` for details.
  These models were trained on VOC 2007 trainval.
    
### Beyond the demo: installation for training and testing models
1. Download the [Attributedevkit](http://10.214.143.222:5000). The directory is */disk50/graduate-doc/zengjiajian/Dataset*.

2. I'll upload the devkit to Google Drive in the near future for your convenience.

3. Extract the tar file into one directory named `Attributedevkit`

	```Shell
	tar xvf Attributedekit.tar
	```
	
4. Create symlinks for the Attributedevkit
	```
	cd $CFA_ROOT/data
	ln -s $Attributedevkit CelebAdevkit
	```
	
	
5. Follow the next sections to download pre-trained ImageNet models

### Download pre-trained ImageNet models

```Shell
cd $FRCN_ROOT
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA by the author of *Faster* R-CNN.

### Usage

Common command for training or test is as follows:

```Shell
cd $CFA_ROOT
./experiments/scripts/plain_zf_for_face_attributes.sh 0 ZF celeba
# *0* is the GPU id
# *ZF* is the network arch which is in {ZF, VGG_CNN_M_1024, VGG16}
# *celeba* is the dataset for training or testing
```

Trained networks are saved under:

```
output/<experiment directory>/<dataset name>/
```

Test outputs are saved under:

```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```
