import functools
import os

import cv2
import numpy as np
import tensorflow as tf

from .util.file import load_file
from .util.xml import xml2dict
from tfdet.core.util import pipeline

LABEL = ["bg", #background
         "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
         "cat", "chair", "cow", "diningtable", "dog", "horse",
         "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
         "tvmonitor"]

COLOR = [(0, 0, 0),
         (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
         (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
         (153, 69, 1), (120, 166, 157), (0, 182, 199), (0, 226, 252),
         (182, 182, 255), (0, 0, 230), (220, 20, 60), (163, 255, 0),
         (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88)]

def load_pascal_voc(path, img_path = None, anno_path = None, mask = False, mask_path = None, only_mask_exist = False):
    """
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012
    
    <example>
    path = "./VOC2007/ImageSets/Main/train.txt"
    img_path = None or "./VOC2007/JPEGImages"
    anno_path = None or "./VOC2007/Annotations"
    mask = with mask_true
    mask_path = None or "./VOC2007/SegmentationObject" or "./VOC2007/SegmentationClass", default path is "SegmentationObject"
    """
    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path)))), "JPEGImages") if img_path is None else img_path
    anno_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path)))), "Annotations") if anno_path is None else anno_path
    mask_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path)))), "SegmentationObject") if mask_path is None else mask_path
    
    pascal_voc = load_file(path)
    for filename in pascal_voc:
        x_true = os.path.join(img_path, "{0}.jpg".format(filename))
        y_true = os.path.join(anno_path, "{0}.xml".format(filename))
        y_true, bbox_true = load_annotation(y_true)
        mask_true = os.path.join(mask_path, "{0}.png".format(filename))
        if only_mask_exist and not os.path.exists(mask_true):
            continue
        if mask:
            if os.path.exists(mask_true):
                mask_true = cv2.imread(mask_true, -1)
            else:
                h, w = np.shape(cv2.imread(x_true, -1))[:2]
                mask_true = np.zeros((h, w, 3), dtype = np.float32)
        result = (x_true, y_true, bbox_true, mask_true) if mask else (x_true, y_true, bbox_true)
        yield result
        
def load_pascal_voc_pipe(path, img_path = None, anno_path = None, mask = False, mask_path = None, only_mask_exist = False,
                         batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
                         cache = None, num_parallel_calls = None):
    generator = functools.partial(load_pascal_voc, path, img_path = img_path, anno_path = anno_path, mask = mask, mask_path = mask_path, only_mask_exist = only_mask_exist)
    dtype = (tf.string, tf.string, tf.int32, tf.float32) if mask else (tf.string, tf.string, tf.int32)
    pipe = tf.data.Dataset.from_generator(generator, dtype)
    return pipeline(pipe, batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                    cache = cache, num_parallel_calls = num_parallel_calls)

def load_annotation(path, bbox = None):
    """
    <example>
    path = "./abc.xml"
    """
    label = path
    if isinstance(path, str):
        anno = xml2dict(path)["annotation"]
        if "object" in anno:
            objs = anno["object"]
            label = []
            bbox = []
            for obj in objs if isinstance(objs, list) else [objs]:
                label.append([obj["name"]])
                bbox.append([int(round(float(obj["bndbox"][k]))) for k in ["xmin", "ymin", "xmax", "ymax"]])
            label = np.array(label, dtype = str)
            bbox = np.array(bbox, dtype = int)
        else:
            label = np.zeros([0, 1], dtype = str)
            bbox = np.zeros([0, 4], dtype = int)
    result = [v for v in [label, bbox] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result
    
def convert_format(path, y_true, bbox_true):
    y_true = np.squeeze(y_true)
    bbox_true = np.squeeze(bbox_true)
    if np.ndim(y_true) == 0:
        y_true = [y_true]
        bbox_true = [bbox_true]
    h, w, c = np.shape(cv2.imread(path))
    data = {"annotation": {"folder": os.path.basename(os.path.dirname(path)),
                           "filename": os.path.basename(path),
                           "path": path,
                           "source": {"database": "Unknown"},
                           "size": {"width": str(w), "height": str(h), "depth": str(c)},
                           "segmented": "0",
                           "object":obj}}
    try:
        if np.max(bbox_true) <= 1:
            bbox_true = np.multiply(bbox_true, [w, h, w, h]).astype(np.int32)
        obj = [{"name": str(y_true[index]),
                "pose": "Unspecified",
                "truncated": "0",
                "difficult": "0",
                "bndbox": {"xmin": str(bbox_true[index][0]), "ymin": str(bbox_true[index][1]), "xmax": str(bbox_true[index][2]), "ymax": str(bbox_true[index][3])}}
               for index in range(len(y_true))]
        if len(obj) == 1:
            obj = obj[0]
        data["annotation"]["object"] = obj
    except:
        pass
    return data