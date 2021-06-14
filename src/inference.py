import os
import sys
import json
import random
import argparse

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures.instances import Instances

from utils import ImageProcessor
from models.config import ModelConfig


def create_arg_parser():
    parser = argparse.ArgumentParser(description='')
    tmp = parser.add_argument_group('Group of ')
    tmp.add_argument('--debug', action='store_true')
    return parser


class Predictor(object):
    def __init__(self):
        self.proc = ImageProcessor()
        self.cfg = ModelConfig(debug=True).cfg
        self.predictor = DefaultPredictor(self.cfg)

    def __call__(self, path='imgs/tmp/sample.jpg') -> Instances:
        image = self.proc(path)
        return self.predictor(image)

    def visualize(self, path, fo='imgs/tmp/_sample.jpg'):
        image = self.proc(path)
        outputs = self.predictor(image)
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.axis('off')
        plt.savefig(fo, bbox_inches='tight')
        print(f'| WRITE ... {fo}')


def debug(args):
    predictor = Predictor()
    outputs = predictor('imgs/tmp/sample.jpg')
    # predictor.visualize('imgs/sample.jpg', fo='imgs/_sample.jpg')


if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()

    if args.debug:
        run(args)
    else:
        debug(args)