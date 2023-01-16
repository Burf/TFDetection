import functools
import os

import cv2
import numpy as np
import tensorflow as tf

from .dataset import Dataset
from .util.file import load_json
from tfdet.core.util import pipeline, py_func

LABEL = ["background", #background
         "balloon"]

memory = {}

def clear(path = None):
    global memory
    if isinstance(path, str):
        key = os.path.basename(path)
        if key in memory:
            del memory[key]
    else:
        memory.clear()
    return memory

def get(path, refresh = False):
    global memory
    key = os.path.abspath(path)
    if refresh or key not in memory:
        balloon = load_json(path)
        balloon = list(balloon.values())
        memory[key] = balloon
    else:
        balloon = memory[key]
    return balloon

def load_info(path, x_true):
    balloon = get(path)
    data_path = os.path.dirname(os.path.abspath(path))
    anno = balloon[x_true]
    
    x_true = os.path.join(data_path, anno["filename"])
    y_true = [r["shape_attributes"] for r in (anno["regions"].values() if isinstance(anno["regions"], dict) else anno["regions"])]
    return x_true, y_true
    
def load_annotation(x_true, y_true, mask = False):
    try:
        import skimage.draw
    except Exception as e:
        print("If you want to use 'balloon dataset', please install 'skimage'")
        raise e
        
    h, w = np.shape(cv2.imread(x_true, -1))[:2]
    new_y_true = np.array([["balloon"]] * len(y_true), dtype = np.object0)
    bbox_true = []
    mask_true = np.zeros((len(y_true), h, w, 1), dtype = np.uint8)
    for i, poly in enumerate(y_true):
        rr, cc = skimage.draw.polygon(poly['all_points_y'], poly['all_points_x'])
        rr = np.clip(rr, 0, h - 1)
        cc = np.clip(cc, 0, w - 1)
        mask_true[i, rr, cc, 0] = 1
        pos = np.where(0 < mask_true[i, ..., 0])[:2]
        bbox = [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])]
        bbox_true.append(bbox)
    y_true = new_y_true
    bbox_true = np.array(bbox_true, dtype = np.int32)
    return (x_true, y_true, bbox_true, mask_true) if mask else (x_true, y_true, bbox_true)

def load_object(path, x_true, mask = False):
    x_true, y_true = load_info(path, x_true)
    return load_annotation(x_true, y_true, mask = mask)

def load_dataset(path, mask = False,
                 transform = None, refresh = False, shuffle = False,
                 cache = None):
    """
    https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
 
    <example>
    1. all-in-one
    > dataset = tfdet.dataset.baloon.load_dataset("./balloon/train/via_region_data.json",
                                                  transform = [load, resize,
                                                               filter_annotation, label_encode, normalize]
                                                  mask = False,
                                                  shuffle = False, cache = "balloon_train.cache")
    > dataset[i] #or next(iter(dataset))
    
    2. split
    > dataset = tfdet.dataset.baloon.load_dataset("./balloon/train/via_region_data.json",
                                                  mask = False,
                                                  shuffle = False, cache = "balloon_train.cache")
    > dataset = tfdet.dataset.Dataset(dataset,
                                      transform = [load, resize,
                                                   filter_annotation, label_encode, normalize])
    > dataset[i] #or next(iter(dataset))
        
    3. dataset to pipe
    > pipe = tfdet.dataset.PipeLoader(dataset)
    > pipe = tfdet.dataset.pipeline.args2dict(pipe) #optional for object detection
    > pipe = tfdet.dataset.pipeline.collect(pipe) #optional for semantic segmentation
    > pipe = tfdet.dataset.pipeline.cast(pipe)
    > pipe = tfdet.dataset.pipeline.key_map(pipe, batch_size = 16, shuffle = False, prefetch = True)
    > next(iter(dataset))
    """
    if isinstance(cache, str) and os.path.exists(cache):
        return Dataset(transform = transform, shuffle = shuffle, cache = cache, keys = ["x_true", "y_true", "bbox_true", "mask_true"])
    else:
        balloon = get(path, refresh = refresh)
        indices = np.arange(len(balloon))
        object_func = functools.partial(load_object, path = path, mask = mask)
        return Dataset(indices, preprocess = object_func, transform = transform, shuffle = shuffle, cache = cache, keys = ["x_true", "y_true", "bbox_true", "mask_true"])