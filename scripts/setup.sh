#!/usr/bin/bash
# Reference: https://detectron2.readthedocs.io/en/latest/tutorials/install.html

cat << END > requirements.in
torch==1.8.1
torchvision==0.9.1
opencv-python==4.5.2.54
ninja==1.10.0.post2
pyyaml==5.4.1
ipdb==0.13.9
gpustat==0.6.0
END

pip install pip-tools
pip-compile requirements.in
pip-sync


if [ ! -f detectron2 ] ; then
    git clone https://github.com/facebookresearch/detectron2.git
fi

python -m pip install -e detectron2
