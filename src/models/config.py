from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode


class ModelConfig(object):
    def __init__(self):
        self.cfg: CfgNode = get_cfg()

    def get_from_model(self, model, fi_yml):
        # model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        return model.get_config_file(fi_yml)

    def set(self, cfg):
        self.cfg.merge_from_file(cfg)

    def __call__(self):
        return self.cfg



