from .effdet import effdet, effdet_d0, effdet_d1, effdet_d2, effdet_d3, effdet_d4, effdet_d5, effdet_d6, effdet_d7, effdet_d7x
from .effdet import effdet_lite, effdet_lite_d0, effdet_lite_d1, effdet_lite_d2, effdet_lite_d3, effdet_lite_d3x, effdet_lite_d4
from .fcos import fcos
from .rcnn import rcnn, faster_rcnn, mask_rcnn, cascade_rcnn, hybrid_task_cascade_rcnn
from .retina import retinanet
from .yolo import yolo_v3, yolo_tiny_v3, yolo_v4, yolo_tiny_v4
hybrid_task_cascade = hybrid_task_cascade_rcnn

from .spade import spade
from .padim import padim
from .patch_core import patch_core

from .deeplab import deeplab_v3, deeplab_v3_plus
from .fcn import fcn
from .pspnet import pspnet
from .unet import unet, unet_2plus
from .upernet import upernet