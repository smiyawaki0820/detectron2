# Detectron2
Facebook AI が提供している Pytorch ベースの物体検出ライブラリ

- https://github.com/facebookresearch/detectron2
- https://detectron2.readthedocs.io/en/latest/tutorials

## Requirements

```txt
gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
```

## Setup
- https://detectron2.readthedocs.io/en/latest/tutorials/install.html

```bash
$ bash scripts/setup.sh
```

### Getting Started

```bash
$ python src/inference.py --debug
```

## Train on a custom dataset

### Pretrained Models
- https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md


### Train

```bash
$ python src/train_coco.py --debug
$ tensorboard --bind_all --logdir outputs
```


## References

### Detectron
- [Configs](https://detectron2.readthedocs.io/en/latest/tutorials/configs.html)
- [Use Custom Datasets](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html)
- [Use Custom Dataloaders](https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html)
- [Use Models](https://detectron2.readthedocs.io/en/latest/tutorials/models.html)
- [Write Models](https://detectron2.readthedocs.io/en/latest/tutorials/write-models.html)
- [Traing (Trainer)](https://detectron2.readthedocs.io/en/latest/tutorials/training.html)

### Others
- [How to Train Detectron2 on Custom Object Detection Data](https://towardsdatascience.com/how-to-train-detectron2-on-custom-object-detection-data-be9d1c233e4)