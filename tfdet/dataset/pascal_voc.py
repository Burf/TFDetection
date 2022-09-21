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
         (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
         (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0),
         (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
         (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0),
         (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]

def load_data(path, mask = False, truncated = True, difficult = False, instance = True, shuffle = False):
    """
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012
    
    <example>
    path = "./VOC2007/ImageSets/Main/train.txt"
    mask = with mask_true
    instance = with instance mask_true
    """
    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path)))), "JPEGImages")
    anno_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path)))), "Annotations")
    mask_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path)))), "SegmentationObject" if instance else "SegmentationClass")
    
    pascal_voc = load_file(path)
    if shuffle:
        np.random.shuffle(pascal_voc)
    for filename in pascal_voc:
        x_true = os.path.join(img_path, "{0}.jpg".format(filename))
        y_true = os.path.join(anno_path, "{0}.xml".format(filename))
        y_true, bbox_true, flag = load_annotation(y_true, truncated = truncated, difficult = difficult, flag = True)
        mask_true = os.path.join(mask_path, "{0}.png".format(filename))
        if mask:
            if not os.path.exists(mask_true):
                continue
            if instance:
                mask_true = load_instance(mask_true)[flag]
            else:
                remove_mask = load_instance(mask_true.replace("SegmentationClass", "SegmentationObject"))[~flag]
                mask_true = load_mask(mask_true)
                for m in remove_mask:
                    mask_true[np.greater(m, 0.5)] = 0
        result = (x_true, y_true, bbox_true, mask_true) if mask else (x_true, y_true, bbox_true)
        yield result
        
def load_pipe(path, mask = False, truncated = True, difficult = False, instance = True, shuffle = False,
              batch_size = 0, epoch = 1, prefetch = False, shuffle_size = None, prefetch_size = None,
              cache = None, num_parallel_calls = None):
    """
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012
    
    <example>
    path = "./VOC2007/ImageSets/Main/train.txt"
    mask = with mask_true
    instance = with instance mask_true
    """
    generator = functools.partial(load_data, path, mask = mask, truncated = truncated, difficult = difficult, instance = instance, shuffle = shuffle and shuffle_size is None)
    dtype = (tf.string, tf.string, tf.int32, tf.float32) if mask else (tf.string, tf.string, tf.int32)
    pipe = tf.data.Dataset.from_generator(generator, dtype)
    return pipeline(pipe, batch_size = batch_size, epoch = epoch, shuffle = shuffle and shuffle_size is not None, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                    cache = cache, num_parallel_calls = num_parallel_calls)

def load_annotation(path, bbox = None, truncated = True, difficult = False, flag = False):
    """
    <example>
    path = "./abc.xml"
    """
    label = path
    if isinstance(path, str):
        anno = xml2dict(path)["annotation"]
        label = []
        bbox = []
        flags = []
        if "object" in anno:
            objs = anno["object"]
            for obj in objs if isinstance(objs, list) else [objs]:
                if not truncated and "truncated" in obj and eval(obj["truncated"]):
                    flags.append(False)
                    continue
                if not difficult and "difficult" in obj and eval(obj["difficult"]):
                    flags.append(False)
                    continue
                flags.append(True)
                label.append([obj["name"]])
                bbox.append([int(round(float(obj["bndbox"][k]))) for k in ["xmin", "ymin", "xmax", "ymax"]])
        label = np.array(label) if 0 < len(label) else np.zeros((0, 1), dtype = str)
        bbox = np.array(bbox) if 0 < len(bbox) else np.zeros((0, 4), dtype = int)
        flags = np.array(flags) if 0 < len(flags) else np.zeros((0,), dtype = np.bool)
    result = [v for v in [label, bbox] if v is not None]
    if flag:
        result += [flags]
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

def load_mask(path, void = False):
    mask = path
    if isinstance(path, str):
        try:
            from PIL import Image
        except Exception as e:
            print("If you want to use 'load_mask', please install 'pillow'")
            raise e
        mask = np.expand_dims(np.array(Image.open(path)), axis = -1)
        if not void:
            mask[mask == 255] = 0
    return mask

def load_instance(path):
    mask_true = load_mask(path)
    if np.ndim(mask_true) < 4:
        h, w = np.shape(mask_true)[:2]
        new_mask_true = []
        for cls in sorted(np.unique(mask_true))[1:]:
            new_mask = np.zeros((h, w, 1), dtype = np.float32)
            new_mask[mask_true == cls] = 1
            new_mask_true.append(new_mask)
        mask_true = np.array(new_mask_true) if 0 < len(new_mask_true) else np.zeros((0, h, w, 1))
    return mask_true

def load_semantic(path):
    return load_mask(path)