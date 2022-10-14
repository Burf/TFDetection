import functools
import os

import cv2
import numpy as np
import tensorflow as tf

from .util.file import load_json
from tfdet.core.util import pipeline

LABEL = ["background", #background
         "balloon"]

def load_data(path, mask = False, shuffle = False):
    """
    https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
    
    <example>
    path = "./balloon/train/via_region_data.json"
    mask = with instance mask_true
    """
    try:
        import skimage.draw
    except Exception as e:
        print("If you want to use 'load_balloon', please install 'skimage'")
        raise e
    
    dir_path = os.path.dirname(os.path.abspath(path))
    balloon = load_json(path)
    balloon = list(balloon.values())
    if shuffle:
        np.random.shuffle(balloon)
    for anno in balloon:
        x_true = os.path.join(dir_path, anno["filename"])
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
        result = (x_true, y_true, bbox_true, mask_true) if mask else (x_true, y_true, bbox_true)
        yield result
        
def load_pipe(path, mask = False, shuffle = False,
              batch_size = 0, repeat = 1, prefetch = False,
              cache = None, num_parallel_calls = None):
    """
    https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
    
    <example>
    path = "./balloon/train/via_region_data.json"
    mask = with instance mask_true
    """
    generator = functools.partial(load_data, path, mask = mask, shuffle = shuffle)
    dtype = (tf.string, tf.string, tf.int32, tf.float32) if mask else (tf.string, tf.string, tf.int32)
    pipe = tf.data.Dataset.from_generator(generator, dtype)
    return pipeline(pipe, batch_size = batch_size, repeat = repeat, shuffle = False, prefetch = prefetch,
                    cache = cache, num_parallel_calls = num_parallel_calls)