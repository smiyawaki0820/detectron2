#!/usr/bin/bash
# USAGE: bash $0

set -ex
source scripts/setting.sh


echo -e "${GREEN}=== Download the balloon segmentation dataset ===${END}"
# https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon

if [ ! -d $DIR_DATA/balloon ] ; then
    mkdir -p $DIR_DATA
    wget -nc https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip -P $DIR_DATA
    unzip $DIR_DATA/balloon_dataset.zip -d $DIR_DATA
    rm $DIR_DATA/balloon_dataset.zip
fi