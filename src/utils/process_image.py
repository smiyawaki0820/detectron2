import os
import requests

import cv2
import numpy as np
from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms


class ImageProcessor(object):
    def __init__(self):
        pass

    def __call__(self, path):
        if os.path.isfile(path):
            # image = Image.open(path)
            image = cv2.imread(path)
        elif path.startswith('http'):
            # image = Image.open(requests.get(path, stream=True).raw)
            image = cv2.imread(requests.get(path, stream=True).raw)
        return image

    @property
    def transform_fn(self):  # 入力画像の処理
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize((.485, .456, .406), (.229, .224, .225))
        ])

    def show(self, path, fo='tmp/tmp.jpg'):
        image = self.__call__(path)
        plt.imshow(image)
        plt.axis('off')
        os.makedirs(os.path.dirname(fo), exist_ok=True)
        plt.savefig(fo, bbox_inches='tight')
        print(f'| WRITE ... {fo}')
