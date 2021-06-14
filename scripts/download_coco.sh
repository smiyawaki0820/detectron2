#! /usr/bin/bash
# USAGE: bash $0

set -ex
BASE=https://dl.fbaipublicfiles.com/detectron2

DEST="datasets/coco/annotations"
mkdir -p $DEST

wget -nc "https://dl.fbaipublicfiles.com/detectron2/annotations/coco/instances_minival2014_100.json" -P $DEST
