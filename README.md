# RetinaNet
An implementation of [RetinaNet](https://arxiv.org/abs/1708.02002) in [PyTorch](http://pytorch.org/).

![RetinaNet Structure](/images/retinanet.png)

* [Installation](#installation)
* [Training](#training)
    * [COCO 2017](#coco-2017)
    * [Pascal VOC](#pascal-voc)
    * [Custom Dataset](#custom-dataset)
* [Evaluation](#evaluation)
* [Todo](#todo)
* [Credits](#credits)

## Installation

1. Install [PyTorch](http://pytorch.org/) and [torchvision](https://github.com/pytorch/vision). 
2. For faster data augmentation, install [pillow-simd](https://github.com/uploadcare/pillow-simd):

```
pip uninstall -y pillow
pip install pillow-simd
```

## Training

### [COCO 2017](http://cocodataset.org/)

1. First, install [pycocotools](https://github.com/pdollar/coco/):

```bash
git clone https://github.com/pdollar/coco/
cd coco/PythonAPI
make
python setup.py install
cd ../..
rm -r coco
```

2. Then download [COCO 2017](http://cocodataset.org/dataset.htm#overview) into `./datasets/COCO/`:

```bash
cd datasets
mkdir COCO
cd COCO
```

If your using `wget`:
```bash
wget http://images.cocodataset.org/zips/train2017.zip &&
wget http://images.cocodataset.org/zips/val2017.zip &&
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

If your using `aria2c` (recommended on for higher bandwidth connections and for allowing resumption of the download.
Tune the number of max concurrent downloads (`-j`) and max connections per server (`-x`) as needed:
```bash
aria2c -x 10 -j 10 http://images.cocodataset.org/zips/train2017.zip &&
aria2c -x 10 -j 10 http://images.cocodataset.org/zips/val2017.zip &&
aria2c -x 10 -j 10 http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip *.zip
rm *.zip
```

Then just run:

```
python train_coco.py
```

### [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/index.html)

```bash
cd datasets
mkdir VOC
cd VOC
```

```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar &&
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar &&
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```

If your using `aria2c` (recommended on for higher bandwidth connections and for allowing resumption of the download.
Tune the number of max concurrent downloads (`-j`) and max connections per server (`-x`) as needed:

```bash
aria2c -x 10 -j wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar &&
aria2c -x 10 -j wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar &&
aria2c -x 10 -j wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

tar xf *.tar
rm *.tar
```

Then just run:

```bash
python train_voc.py
```

### Custom Dataset
Lots to write here. :wink:

## Evaluation
To evaluate an image on a trained model:
```
python eval.py [checkpoint_path] [image_path]
```
This will create an image (`output.jpg`) with bounding box annotations.

## Todo

1. Finish converting the COCO dataset class to work with batches.
2. Train [COCO 2017](http://cocodataset.org/) for 90,000 iterations and save a reusable checkpoint.
3. Try training on [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and add download instructions.
4. Produce bounding box outputs for a few sanity check images.
5. Upload trained weights to Github releases.
5. Train on the 🔮magic proprietary dataset ✨. 

## Credits
