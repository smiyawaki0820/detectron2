import os

from detectron2.config import get_cfg, CfgNode
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog

from data.balloon import get_balloon_dicts


DDIR="/work02/miyawaki/exp2021/detectron2/data"

def create_cfg():
    cfg: CfgNode = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (f"balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    return cfg


def run():
    cfg = create_cfg()
    with open('configs/ft_balloon_base.yml', 'w') as fo:
        fo.write(cfg.dump())
        print(f'| WRITE ... {fo.name}')

    for d in ["train"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts(f"{DDIR}/balloon/{d}"))
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
    balloon_metadata = MetadataCatalog.get("balloon_train")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__':
    run()