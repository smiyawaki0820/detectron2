import os
import sys
import json
import argparse

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json



def create_arg_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dest', default='outputs', type=str, help='dir_out')

    data = parser.add_argument_group('Group of Dataset')
    data.add_argument('--dir_img_train', default='/work02/miyawaki/exp2021/CLIP/datasets/coco/train2014', type=str, help='')
    data.add_argument('--dir_img_valid', default='/work02/miyawaki/exp2021/CLIP/datasets/coco/valid2014', type=str, help='')
    data.add_argument('--fi_detect_train', default='/work02/miyawaki/exp2021/CLIP/datasets/coco/annotations/instances_train2014.json', type=str, help='')
    data.add_argument('--fi_detect_valid', default='/work02/miyawaki/exp2021/CLIP/datasets/coco/annotations/instances_val2014.json', type=str, help='')
    data.add_argument('--data_name_train', default='coco_train2014', type=str, help='')
    data.add_argument('--data_name_valid', default='coco_val2014', type=str, help='')

    model = parser.add_argument_group('Group of Model')
    model.add_argument('--fi_cfg', default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml', type=str, help='https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md')
    model.add_argument('--fi_ckpt', default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml', type=str, help='https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md')
    
    hparam = parser.add_argument_group('Group of HyperParameters')
    hparam.add_argument('--n_workers', default=2, type=int, help='n_workers of dataloader')
    hparam.add_argument('--learning_rate', default=0.00025, type=float)
    hparam.add_argument('--max_iter', default=300, type=float)
    hparam.add_argument('--batch_size', default=128, type=int)
    hparam.add_argument('--ims_per_batch', default=2, type=int, help='If we use 16 GPUs and IMS_PER_BATCH = 32, each GPU will see 2 images per batch.')

    return parser


def create_cfg(args) -> CfgNode:
    # https://detectron2.readthedocs.io/en/latest/modules/config.html#config-references
    cfg = get_cfg()
    cfg.OUTPUT_DIR = args.dest
    cfg.merge_from_file(model_zoo.get_config_file(args.fi_cfg))
    cfg.DATASETS.TRAIN = (args.data_name_train, args.data_name_valid)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = args.n_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.fi_ckpt)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.n_classes
    if args.debug:
        cfg.DATASETS.TRAIN = (args.data_name_valid)
        cfg.SOLVER.MAX_ITER = 3
    return cfg


def run():
    parser = create_arg_parser()
    args = parser.parse_args()
    os.makedirs(args.dest, exist_ok=True)

    register_coco_instances(args.data_name_train, {}, args.fi_detect_train, args.dir_img_train)
    register_coco_instances(args.data_name_valid, {}, args.fi_detect_valid, args.dir_img_valid)

    data_train = DatasetCatalog.get(args.data_name_train)
    data_valid = DatasetCatalog.get(args.data_name_valid)
    metadata_train = MetadataCatalog.get(args.data_name_train)
    metadata_valid = MetadataCatalog.get(args.data_name_valid)
    assert metadata_train.thing_classes == metadata_valid.thing_classes
    
    args.n_classes = len(metadata_train.thing_classes) + 1  # 背景

    # data_dict = load_coco_json(args.fi_detect_valid, args.dir_img_valid, dataset_name=args.data_name_valid, extra_annotation_keys=None)

    cfg = create_cfg(args)

    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__':
    run()