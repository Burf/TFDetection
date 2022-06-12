from .fcos import fcos_head
from .rcnn import rpn_head, Rpn2Proposal, faster_rcnn_head, mask_rcnn_head, cascade_rcnn_head, hybrid_task_cascade_rcnn_head
from .retina import retina_head
from .yolo import yolo_v3_head, yolo_tiny_v3_head, yolo_v4_head, yolo_tiny_v4_head

from .spade import spade_head
from .padim import padim_head
from .patch_core import patch_core_head

from .deeplab import deeplab_v3_head
from .fcn import fcn_head
from .pspnet import pspnet_head
from .unet import unet_head, unet_2plus_head
from .upernet import upernet_head