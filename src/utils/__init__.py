import os
import sys
sys.path.append('src/utils')

# Logging
from detectron2.utils.logger import setup_logger
setup_logger()

from process_image import ImageProcessor
