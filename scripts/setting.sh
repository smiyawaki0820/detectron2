#!/usr/bin/bash
# USAGE=". $0"

DATE=`date +%Y%m%d-%H%M`

ROOT=`pwd`
export PYTHONPATH=$ROOT:$PYTHONPATH

END="\e[m"
GREEN="\e[32m"
BLUE="\e[34m"
YELLOW="\e[33m"


DEST="/work02/miyawaki/exp2021/detectron2"
DIR_DATA="$DEST/data"
DIR_MODEL="$DEST/models"