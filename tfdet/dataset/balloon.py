import functools
import os

import cv2
import numpy as np
import tensorflow as tf

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

def load_data(path, mask = False, refresh = False, shuffle = False):
    """
    https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
    
    <example>
    path = "./balloon/train/via_region_data.json"
    mask = with instance mask_true
    """
    data_path = os.path.dirname(os.path.abspath(path))
    balloon = get(path, refresh = refresh)
    if shuffle:
        np.random.shuffle(balloon)
    for anno in balloon:
        yield load_object(data_path, anno, mask = mask)
        
def load_object(data_path, anno, mask = False):
    try:
        import skimage.draw
    except Exception as e:
        print("If you want to use 'balloon dataset', please install 'skimage'")
        raise e
        
    x_true = os.path.join(data_path, anno["filename"])
    h, w = np.shape(cv2.imread(x_true, -1))[:2]
    polygons = [r['shape_attributes'] for r in (anno["regions"].values() if isinstance(anno["regions"], dict) else anno["regions"])]
    y_true = np.array([["balloon"]] * len(polygons), dtype = str)
    bbox_true = []
    mask_true = np.zeros((len(polygons), h, w, 1), dtype = np.float32)
    for i, poly in enumerate(polygons):
        rr, cc = skimage.draw.polygon(poly['all_points_y'], poly['all_points_x'])
        rr = np.clip(rr, 0, h - 1)
        cc = np.clip(cc, 0, w - 1)
        mask_true[i, rr, cc, 0] = 1
        pos = np.where(0 < mask_true[i, ..., 0])[:2]
        bbox = [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])]
        bbox_true.append(bbox)
    bbox_true = np.array(bbox_true, dtype = int)
    return (x_true, y_true, bbox_true, mask_true) if mask else (x_true, y_true, bbox_true)

def load_index(path, index, mask = False):
    balloon = get(path)
    data_path = os.path.dirname(os.path.abspath(path))
    return load_object(data_path, balloon[index], mask = mask)

def load_pipe(path, mask = False, refresh = False, shuffle = False,
              batch_size = 0, repeat = 1, prefetch = False,
              cache = None, num_parallel_calls = True):
    """
    https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
    
    <example>
    path = "./balloon/train/via_region_data.json"
    mask = with instance mask_true
    """
    balloon = get(path, refresh = refresh)
    indices = np.arange(len(balloon))
    if shuffle:
        np.random.shuffle(indices)
    
    object_func = functools.partial(load_index, path, mask = mask)
    dtype = (tf.string, tf.string, tf.int32, tf.float32) if mask else (tf.string, tf.string, tf.int32)
    func = functools.partial(py_func, object_func, Tout = dtype)
    return pipeline(indices, function = func,
                    batch_size = batch_size, repeat = repeat, shuffle = False, prefetch = prefetch,
                    cache = cache, num_parallel_calls = num_parallel_calls)

def load_pipe_old(path, mask = False, refresh = False, shuffle = False,
                  batch_size = 0, repeat = 1, prefetch = False,
                  cache = None, num_parallel_calls = True):
    """
    https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
    
    <example>
    path = "./balloon/train/via_region_data.json"
    mask = with instance mask_true
    """
    generator = functools.partial(load_data, path, mask = mask, refresh = refresh, shuffle = shuffle)
    dtype = (tf.string, tf.string, tf.int32, tf.float32) if mask else (tf.string, tf.string, tf.int32)
    pipe = tf.data.Dataset.from_generator(generator, dtype)
    return pipeline(pipe, batch_size = batch_size, repeat = repeat, shuffle = False, prefetch = prefetch,
                    cache = cache, num_parallel_calls = num_parallel_calls)