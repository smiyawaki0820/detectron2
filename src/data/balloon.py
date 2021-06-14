""" if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")
"""

import os
import sys
import json
import random

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer


def get_balloon_dicts(dir_img):
    dataset_dicts = []
    imgs_anns = json.load(open(os.path.join(dir_img, "via_region_data.json")))
    
    for idx, v in enumerate(imgs_anns.values()):
        filename = os.path.join(dir_img, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            objs.append({
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            })

        dataset_dicts.append({
            "file_name": filename,
            "image_id": idx,
            "height": height,
            "width": width,
            "annotations": objs
        })
        
    return dataset_dicts


if __name__ == '__main__':
    """ run
    $ python $0
    """

    DDIR="/work02/miyawaki/exp2021/detectron2/data"

    for d in ["val"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts(f"{DDIR}/balloon/{d}"))
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
    balloon_metadata = MetadataCatalog.get("balloon_train")

    dataset_dicts = get_balloon_dicts(f"{DDIR}/balloon/train")
    for d in random.sample(dataset_dicts, 1):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        plt.imshow(out.get_image()[:, :, ::-1])
        fo_img = 'imgs/tmp/balloon.jpg'
        plt.savefig(fo_img, bbox_inches='tight')
        print(f'| WRITE ... {fo_img}')


    import ipdb; ipdb.set_trace()