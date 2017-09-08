# RetinaNet
An implementation of [RetinaNet](https://arxiv.org/abs/1708.02002).

![RetinaNet Structure](/images/retinanet.png)

## Installation

1. Install [PyTorch](http://pytorch.org/) and [torchvision](https://github.com/pytorch/vision). 

2. Install `pycocotools`:

```
git clone https://github.com/pdollar/coco/
cd coco/PythonAPI
make
python setup.py install
```

3. Download [COCO 2017](http://cocodataset.org/dataset.htm#overview) into `./datasets/COCO/`.
