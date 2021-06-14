# Tutorial for Detectron2
- [Colab Tutorial](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)

## Config
- [Docs/Tutorials/Configs](https://detectron2.readthedocs.io/en/latest/tutorials/configs.html)
- [detectron2.config](https://detectron2.readthedocs.io/en/latest/modules/config.html#detectron2-config)
    - [a list of available configs](https://detectron2.readthedocs.io/en/latest/modules/config.html#config-references)

```python
>>> from detectron2.config import get_cfg, CfgNode

>>> cfg: CfgNode = get_cfg()    # default config
>>> cfg.xxx = yyy               # overwrite
>>> cfg.merge_from_file(fi_cfg) # load config file

>>> with open(fo_config, 'w') as fo_yml:
>>>     fo_yml.write(cfg.dump())
```

コマンドラインからも上書きできる

```bash
$ ./demo.py \
    --config-file config.yaml \
    [--other-options] \
    --opts MODEL.WEIGHTS /path/to/weights \
    INPUT.MIN_SIZE_TEST 1000
```

## Dataset
- [detectron2.data.DatasetCatalog](https://detectron2.readthedocs.io/en/latest/modules/data.html#detectron2.data.DatasetCatalog)
    - To store information about the datasets and how to obtain them.
- [detectron2.data.MetadataCatalog](https://detectron2.readthedocs.io/en/latest/modules/data.html#detectron2.data.MetadataCatalog)

### Use Builtin Datasets
- [Docs/Tutorials]

### Use Custom Datasets
- [Docs/Tutorials/Use Custom Datasets](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html)
- [detectron2.structures.BoxMode](https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.BoxMode)
- [Register a COCO Format Dataset](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#register-a-coco-format-dataset)
- [Update the Config for New Datasets](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#update-the-config-for-new-datasets)

```python
>>> from detectron2.data import DatasetCatalog, MetadataCatalog
>>> def my_dataset_func():
        return List[dict]   # standard/custom dataset dict

{
    # Common fields
    'file_name': str, # fi_image
    'height': int,
    'width': int,
    'image_id': Union[str, int],
    # Instance detection/segmentation
    'annotations': [{   # each dict: one instance
        'bbox': List[float],    # [x1, y1, x2, y2]
        'bbox_mode': int,       # BoxMode.XYXY_ABS, BoxMode.XYWH_ABS
        'category_id': int,
        'segmentation':List[dict],    # segmentation mask
        'keypoints': List[float],
        'iscrowd': int=0
    }],
    # Semantic segmentation
    'sem_seg_file_name': str,
    # Panoptic segmentation
    'pan_seg_file_name': str,
    'segments_info': List[dict],
}

>>> DatasetCatalog.register("my_dataset", my_dataset_function)
>>> data: List[Dict] = DatasetCatalog.get("my_dataset")
>>> metadata = MetadataCatalog.get("my_dataset")
>>> metadata.some_key = some_value
```


## DataLoader
- [Docs/Tutorials/DataLoader](https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html)

## Models
- [Detectron2 Model Zoo and Baselines](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)
- [Projects by Facebook](https://github.com/facebookresearch/detectron2/tree/master/projects)



### Use Models
- [Docs/Tutorials/Use Models](https://detectron2.readthedocs.io/en/latest/tutorials/models.html)

```python
>>> from detectron2.modeling import build_model
>>> model: torch.nn.Module = build_model(cfg)
>>> outputs = model(inputs: List[dict])

# Load
>>> from detectron2.checkpoint import DetectionCheckpointer
>>> DetectionCheckpointer(model).load(path)

# Save
>>> checkpointer = DetectionCheckpointer(model, save_dir=dir_out)
>>> checkpointer.save(fo_name)  # {dir_out}/{fo_name}.pth
```

- [Model Input Format](https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-input-format)
    - [detectron2.structures.Instances](https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances)]
    - [detectron2.structures.Boxes](https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Boxes)
    - [detectron2.structures.Keypoint](https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Keypoints)
    - [detectron2.structures.BitMask](https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.BitMasks)
    - [detectron2.structures.PolygonMasks](https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.PolygonMasks)
- [Model Output Format](https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format)

```
# inputs: List[dict]
{
    "image": Tensor(C, H, W),   # Input
    "height": ,
    "width": ,
    "instances": Instances(
        "gt_boxes": Boxes,      # N Boxes
        "gt_classes": Tensor,   # N labels [0, n_categories)
        "gt_masks": Union[PolygonMasks, BitMasks],  # N masks
        "gt_keypoint": Keypoint,    # N keypoint sets
    ),
    "sem_seg": Tensor(H, W),
    "proposals": Instances(
        "proposal_boxes": Boxes,
        "objectness_logits": Tensor,
    )
}

# Boxes
Tensor(N*(x1, y1, x2, y2))

# Output
```


### Write Models
- [Docs/Tutorials/Write Models](https://detectron2.readthedocs.io/en/latest/tutorials/write-models.html)

## Training
- [Docs/Tutorials/Training](https://detectron2.readthedocs.io/en/latest/tutorials/training.html#training)
- [tools/plain_train_net.py](https://github.com/facebookresearch/detectron2/blob/master/tools/plain_train_net.py)

### Trainer
- [detectron2.engine.SimpleTrainer](https://detectron2.readthedocs.io/en/latest/modules/engine.html#detectron2.engine.SimpleTrainer)
- [detectron2.engine.defaults.DefaultTrainer](https://detectron2.readthedocs.io/en/latest/modules/engine.html#detectron2.engine.defaults.DefaultTrainer)
    - カスタマイズする際は、クラス継承を利用して、対象のメソッドを overrides する（[参考](https://github.com/facebookresearch/detectron2/blob/master/tools/train_net.py)）
    - また [hook system](https://detectron2.readthedocs.io/en/latest/modules/engine.html#detectron2.engine.HookBase) も同様に変更
    - [tools/plain_train_net.py](https://github.com/facebookresearch/detectron2/blob/master/tools/plain_train_net.py) のようにスクラッチを作成する方法も可能

### Logging of Metrics
- [detectron2.utils.events.EventStorage](https://detectron2.readthedocs.io/en/latest/modules/utils.html#detectron2.utils.events.EventStorage)
- [detectron2.utils.events module](https://detectron2.readthedocs.io/en/latest/modules/utils.html#module-detectron2.utils.events)

```python
>>> from detectron2.utils.events import EventStorage
>>> with EventStorage() as storage:
>>>     storage.put_scalar("accuracy", accuracy)
```

## Evaluation
- [Docs/Tutorials/Evaluation](https://detectron2.readthedocs.io/en/latest/tutorials/evaluation.html)