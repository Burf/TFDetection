import functools
import os

import cv2
import numpy as np
import tensorflow as tf

from .dataset import Dataset
from .util.file import load_file
from .util.xml import xml2dict
from tfdet.core.util import pipeline, py_func

LABEL = ["background", #background
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
        
def load_object(x_true, y_true, mask_true = None, truncated = True, difficult = False, instance = True, bbox_true = None):
    if bbox_true is not None and mask_true is None:
        mask_true = bbox_true
    y_true, bbox_true, flag = load_annotation(y_true, truncated = truncated, difficult = difficult, flag = True)
    if mask_true is not None:
        if instance:
            mask_true = load_instance(mask_true)[flag]
        else:
            remove_mask = load_instance(mask_true.replace("SegmentationClass", "SegmentationObject"))[~flag]
            mask_true = load_mask(mask_true)
            for m in remove_mask:
                mask_true[np.greater(m, 0.5)] = 0
    return (x_true, y_true, bbox_true, mask_true) if mask_true is not None else (x_true, y_true, bbox_true)

def load_dataset(path, mask = False, truncated = True, difficult = False, instance = True, 
                 transform = None, shuffle = False,
                 cache = None):
    """
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012
    
    <example>
    1. all-in-one
    > dataset = tfdet.dataset.pascal_voc.load_dataset("./VOC2007/ImageSets/Main/train.txt",
                                                      mask = False, truncated = True, difficult = False, instance = True,
                                                      transform = [{"name":"load"},
                                                                   {"name":"resize", "image_shape":[512, 512]},
                                                                   {"name":"pad", "image_shape":[512, 512]},
                                                                   {"name":"filter_annotation"},
                                                                   {"name":"label_encode", "label":tfdet.dataset.pascal_voc.LABEL},
                                                                   {"name":"normalize", "mean":[123.675, 116.28, 103.53], "std":[58.395, 57.12, 57.375]}]
                                                      shuffle = False, cache = "voc2007_train.cache")
    > dataset[i] #or next(iter(dataset))
    
    2. split
    > dataset = tfdet.dataset.pascal_voc.load_dataset("./VOC2007/ImageSets/Main/train.txt",
                                                      mask = False, truncated = True, difficult = False, instance = True, 
                                                      shuffle = False, cache = "voc2007_train.cache")
    > dataset = tfdet.dataset.Dataset(dataset,
                                      transform = [{"name":"load"},
                                                   {"name":"resize", "image_shape":[512, 512]},
                                                   {"name":"pad", "image_shape":[512, 512]},
                                                   {"name":"filter_annotation"},
                                                   {"name":"label_encode", "label":tfdet.dataset.pascal_voc.LABEL},
                                                   {"name":"normalize", "mean":[123.675, 116.28, 103.53], "std":[58.395, 57.12, 57.375]}])
    > dataset[i] #or next(iter(dataset))
        
    3. dataset to pipe
    > pipe = tfdet.dataset.PipeLoader(dataset)
    > pipe = tfdet.dataset.pipeline.args2dict(pipe) #optional for object detection
    > pipe = tfdet.dataset.pipeline.collect(pipe) #filtered item by key
    > pipe = tfdet.dataset.pipeline.cast(pipe)
    > pipe = tfdet.dataset.pipeline.key_map(pipe, batch_size = 16, shuffle = False, prefetch = True)
    > next(iter(dataset))
    """
    if isinstance(cache, str) and os.path.exists(cache):
        return Dataset(transform = transform, shuffle = shuffle, cache = cache, keys = ["x_true", "y_true", "bbox_true", "mask_true"])
    else:
        img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path)))), "JPEGImages")
        anno_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path)))), "Annotations")
        mask_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path)))), "SegmentationObject" if instance else "SegmentationClass")

        pascal_voc = load_file(path)
        if shuffle:
            np.random.shuffle(pascal_voc)
        x_trues, y_trues, mask_trues = [], [], []
        for filename in pascal_voc:
            x_true = os.path.join(img_path, "{0}.jpg".format(filename))
            y_true = os.path.join(anno_path, "{0}.xml".format(filename))
            mask_true = os.path.join(mask_path, "{0}.png".format(filename)) if mask else None
            if mask and not os.path.exists(mask_true):
                continue
            x_trues.append(x_true)
            y_trues.append(y_true)
            if mask_true is not None:
                mask_trues.append(mask_true)
        args = (np.array(x_trues, dtype = np.object0), np.array(y_trues, dtype = np.object0))
        if 0 < len(mask_trues):
            args += (np.array(mask_trues, dtype = np.object0),)
        object_func = functools.partial(load_object, truncated = truncated, difficult = difficult, instance = instance)
        return Dataset(*args, preprocess = object_func, transform = transform, shuffle = shuffle, cache = cache, keys = ["x_true", "y_true", "bbox_true", "mask_true"])

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
        label = np.array(label, dtype = np.object0) if 0 < len(label) else np.zeros((0, 1), dtype = np.object0)
        bbox = np.array(bbox, dtype = np.int32) if 0 < len(bbox) else np.zeros((0, 4), dtype = np.int32)
        flags = np.array(flags, dtype = bool) if 0 < len(flags) else np.zeros((0,), dtype = bool)
    result = [v for v in [label, bbox] if v is not None]
    if flag:
        result += [flags]
    result = result[0] if len(result) == 1 else tuple(result)
    return result
    
def convert_format(path, y_true, bbox_true, label = None):
    """
    path = image path
    y_true = (N, 1 or n_class)
    bbox_true = (N, 4)
    """
    if np.ndim(y_true) == 3:
        y_true = y_true[0]
        bbox_true = bbox_true[0]
    bbox_true = np.array(bbox_true) if not isinstance(bbox_true, np.ndarray) else bbox_true
    valid_indices = np.where(0 < np.max(bbox_true, axis = -1))
    y_true = (np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true)[valid_indices]
    bbox_true = bbox_true[valid_indices]
    if np.shape(y_true)[-1] != 1:
        y_true = np.expand_dims(np.argmax(y_true, axis = -1), axis = -1)
    if label is not None and np.issubdtype(y_true.dtype, np.number):
        y_true = np.array(label)[y_true]
    
    h, w, c = np.shape(cv2.imread(path))
    data = {"annotation": {"folder": os.path.basename(os.path.dirname(path)),
                           "filename": os.path.basename(path),
                           "path": path,
                           "source": {"database": "Unknown"},
                           "size": {"width": str(w), "height": str(h), "depth": str(c)},
                           "segmented": "0"}}
    try:
        if np.max(bbox_true) < 2:
            bbox_true = np.multiply(bbox_true, [w, h, w, h], dtype = np.float32)
        bbox_true = bbox_true.astype(np.int32)
        obj = [{"name": str(y_true[index][0]),
                "pose": "Unspecified",
                "truncated": "0",
                "difficult": "0",
                "bndbox": {"xmin": str(bbox_true[index][0]), "ymin": str(bbox_true[index][1]), "xmax": str(bbox_true[index][2]), "ymax": str(bbox_true[index][3])}}
               for index in range(len(y_true))]
        data["annotation"]["object"] = obj[0] if len(obj) == 1 else obj
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
        mask = np.expand_dims(np.array(Image.open(path), dtype = np.uint8), axis = -1)
        if not void:
            mask[mask == 255] = 0
    return mask

def load_instance(path):
    mask_true = load_mask(path)
    if np.ndim(mask_true) < 4:
        h, w = np.shape(mask_true)[:2]
        new_mask_true = []
        for cls in sorted(np.unique(mask_true))[1:]:
            new_mask = np.zeros((h, w, 1), dtype = np.uint8)
            new_mask[mask_true == cls] = 1
            new_mask_true.append(new_mask)
        mask_true = np.stack(new_mask_true, axis = 0).astype(np.uint8) if 0 < len(new_mask_true) else np.zeros((0, h, w, 1), dtype = np.uint8)
    return mask_true

def load_semantic(path):
    return load_mask(path)

def convert_tfds_to_tfdet(data, truncated = True, difficult = False, label = LABEL[1:]):
    x_true = data["image"]
    obj = data["objects"]
    y_true = tf.expand_dims(obj["label"], axis = -1)
    if label is not None:
        y_true = tf.gather(label, y_true)
    bbox_true = obj["bbox"]
    diff = obj["is_difficult"]
    if not truncated:
        indices = tf.where(tf.logical_not(obj["is_truncated"]))[:, 0]
        y_true = tf.gather(y_true, indices)
        bbox_true = tf.gather(bbox_true, indices)
        diff = tf.gather(diff, indices)
    if not difficult:
        indices = tf.where(tf.logical_not(diff))[:, 0]
        y_true = tf.gather(y_true, indices)
        bbox_true = tf.gather(bbox_true, indices)
    return x_true, y_true, bbox_true

def tfds_to_tfdet(tfds_pipe, truncated = True, difficult = False, label = LABEL[1:],
                  batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
                  cache = False, num_parallel_calls = True):
    """
    <example>
    import tfdet
    import tensorflow_datasets as tfds
    pipe = tfds.load("voc/2007", split = tfds.Split.VALIDATION)
    pipe = tfdet.dataset.pascal_voc.tfds_to_tfdet(pipe, truncated = True, difficult = False)

    x_true, y_true, bbox_true = next(iter(pipe)) #x_true:(375, 500, 3), y_true:(3, 1), bbox_true:(3, 4)
    """
    func = functools.partial(convert_tfds_to_tfdet, truncated = truncated, difficult = difficult, label = label)
    pipe = tfds_pipe.map(func)
    return pipeline(pipe,
                    batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                    cache = cache, num_parallel_calls = num_parallel_calls)