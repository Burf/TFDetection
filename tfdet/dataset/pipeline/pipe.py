import functools

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf

from tfdet.core.util import pipeline, py_func
from .transform import load, preprocess, resize, pad, crop, random_crop, mosaic, cut_mix, albumentations
from .formatting import key_map, collect
from ..util import load_image, load_pascal_voc

def custom_pipe(function, x_true, y_true = None, bbox_true = None, mask_true = None, dtype = None, tf_func = False,
                pre_batch_size = 0, pre_shuffle = False, pre_shuffle_size = None, pre_prefetch = False, pre_prefetch_size = None,
                batch_size = 0, epoch = 1, shuffle = False, prefetch = False, num_parallel_calls = None, cache = None, shuffle_size = None, prefetch_size = None,
                **kwargs):
    args = [arg for arg in [x_true, y_true, bbox_true, mask_true] if arg is not None]
    args = args[0] if len(args) == 1 else tuple(args)
    pre_pipe = pipeline(args, batch_size = pre_batch_size, shuffle = pre_shuffle, prefetch = pre_prefetch, num_parallel_calls = num_parallel_calls, shuffle_size = pre_shuffle_size, prefetch_size = pre_prefetch_size)
    func = None
    if tf_func:
        func = functools.partial(function, **kwargs)
    elif callable(function):
        if dtype is None:
            if isinstance(x_true, tf.data.Dataset):
                dtype = tuple(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else pre_pipe.element_spec
            else:
                if isinstance(x_true, dict):
                    sample_result = function(**{k:v[0] for k, v in x_true.items()}, **kwargs)
                else:
                    sample_args = [v[0] for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
                    sample_result = function(*sample_args, **kwargs)
                dtype = [tf.convert_to_tensor(r).dtype for r in (sample_result if isinstance(sample_result, tuple) else (sample_result,))]
                dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
        elif np.ndim(dtype) == 0:
            if isinstance(x_true, tf.data.Dataset):
                if isinstance(x_true.element_spec, tuple) or isinstance(x_true.element_spec, dict):
                    dtype = [dtype] * len(x_true.element_spec)
            else:
                dtype = [dtype] * len(x_true if isinstance(x_true, dict) else [arg for arg in [x_true, y_true, bbox_true, mask_true] if arg is not None])
                dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
        func = functools.partial(py_func, function, Tout = dtype, **kwargs) if callable(function) else None
    pipe = pipeline(pre_pipe, map = func, batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)
    return pipe

def load_pipe(x_true, y_true = None, bbox_true = None, mask_true = None, 
              load_func = load_image, anno_func = load_pascal_voc,
              batch_size = 0, epoch = 1, shuffle = False, prefetch = False, num_parallel_calls = None, cache = None, shuffle_size = None, prefetch_size = None):
    pipe = custom_pipe(load, x_true, y_true, bbox_true, mask_true, dtype = None, tf_func = False,
                       load_func = load_func, anno_func = anno_func,
                       batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)
    return pipe

def preprocess_pipe(x_true, y_true = None, bbox_true = None, mask_true = None, 
                    dtype = tf.float32,
                    rescale = 1., mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375],
                    label = None, one_hot = True, label_smoothing = 0.1,
                    bbox_normalize = True, min_area = 0.,
                    batch_size = 0, epoch = 1, shuffle = False, prefetch = False, num_parallel_calls = None, cache = None, shuffle_size = None, prefetch_size = None):
    pipe = custom_pipe(preprocess, x_true, y_true, bbox_true, mask_true, dtype = dtype, tf_func = False,
                       rescale = rescale, mean = mean, std = std,
                       label = label, one_hot = one_hot, label_smoothing = label_smoothing,
                       bbox_normalize = bbox_normalize, min_area = min_area,
                       batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)
    return pipe
    
def resize_pipe(x_true, y_true = None, bbox_true = None, mask_true = None, 
                image_shape = None,
                batch_size = 0, epoch = 1, shuffle = False, prefetch = False, num_parallel_calls = None, cache = None, shuffle_size = None, prefetch_size = None):
    pipe = custom_pipe(resize, x_true, y_true, bbox_true, mask_true, dtype = None, tf_func = False,
                       image_shape = image_shape,
                       batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)
    return pipe

def pad_pipe(x_true, y_true = None, bbox_true = None, mask_true = None, 
             image_shape = None, max_pad_size = 100, pad_val = 0, background = "bg", mode = "right",
             batch_size = 0, epoch = 1, shuffle = False, prefetch = False, num_parallel_calls = None, cache = None, shuffle_size = None, prefetch_size = None):
    pipe = custom_pipe(pad, x_true, y_true, bbox_true, mask_true, dtype = None, tf_func = False,
                       image_shape = image_shape, max_pad_size = max_pad_size, pad_val = pad_val, background = background, mode = mode,
                       batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)
    return pipe

def crop_pipe(x_true, y_true = None, bbox_true = None, mask_true = None, 
              bbox = None, min_area = 0., min_visibility = 0.,
              batch_size = 0, epoch = 1, shuffle = False, prefetch = False, num_parallel_calls = None, cache = None, shuffle_size = None, prefetch_size = None):
    """
    bbox = [x1, y1, x2, y2]
    """
    pipe = custom_pipe(crop, x_true, y_true, bbox_true, mask_true, dtype = None, tf_func = False,
                       bbox = bbox, min_area = min_area, min_visibility = min_visibility,
                       batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)
    return pipe

def random_crop_pipe(x_true, y_true = None, bbox_true = None, mask_true = None,
                     image_shape = None, min_area = 0., min_visibility = 0.,
                     batch_size = 0, epoch = 1, shuffle = False, prefetch = False, num_parallel_calls = None, cache = None, shuffle_size = None, prefetch_size = None):
    pipe = custom_pipe(random_crop, x_true, y_true, bbox_true, mask_true, dtype = None, tf_func = False,
                       image_shape = image_shape, min_area = min_area, min_visibility = min_visibility,
                       batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)
    return pipe
    
def mosaic_pipe(x_true, y_true = None, bbox_true = None, mask_true = None, 
                image_shape = None, alpha = 0.2, pad_val = 0, min_area = 0., min_visibility = 0., e = 1e-12, 
                pre_batch_size = 4, pre_shuffle = False, pre_shuffle_size = None, pre_prefetch = False, pre_prefetch_size = None,
                batch_size = 0, epoch = 1, shuffle = False, prefetch = False, num_parallel_calls = None, cache = None, shuffle_size = None, prefetch_size = None):
    pipe = custom_pipe(mosaic, x_true, y_true, bbox_true, mask_true, dtype = None, tf_func = False,
                       image_shape = image_shape, alpha = alpha, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e,
                       pre_batch_size = pre_batch_size, pre_shuffle = pre_shuffle, pre_shuffle_size = pre_shuffle_size, pre_prefetch = pre_prefetch, pre_prefetch_size = pre_prefetch_size,
                       batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)
    return pipe
    
def cut_mix_pipe(x_true, y_true = None, bbox_true = None, mask_true = None,
                 alpha = 0.2, min_area = 0., min_visibility = 0., e = 1e-12, 
                 pre_batch_size = 2, pre_shuffle = False, pre_shuffle_size = None, pre_prefetch = False, pre_prefetch_size = None,
                 batch_size = 0, epoch = 1, shuffle = False, prefetch = False, num_parallel_calls = None, cache = None, shuffle_size = None, prefetch_size = None):
    pipe = custom_pipe(cut_mix, x_true, y_true, bbox_true, mask_true, dtype = None, tf_func = False,
                       alpha = alpha, min_area = min_area, min_visibility = min_visibility, e = e,
                       pre_batch_size = pre_batch_size, pre_shuffle = pre_shuffle, pre_shuffle_size = pre_shuffle_size, pre_prefetch = pre_prefetch, pre_prefetch_size = pre_prefetch_size,
                       batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)
    return pipe

def albumentations_pipe(x_true, y_true = None, bbox_true = None, mask_true = None,
                        transform = [A.Blur(p = 0.01),
                                     A.MedianBlur(p = 0.01),
                                     A.ToGray(p = 0.01),
                                     A.CLAHE(p = 0.01, clip_limit = 4., tile_grid_size = (8, 8)),
                                     A.RandomBrightnessContrast(p = 0.01, brightness_limit = 0.2, contrast_limit = 0.2),
                                     A.RGBShift(p = 0.01, r_shift_limit = 10, g_shift_limit = 10, b_shift_limit = 10),
                                     A.HueSaturationValue(p = 0.01, hue_shift_limit = 10, sat_shift_limit = 40, val_shift_limit = 50),
                                     A.ChannelShuffle(p = 0.01),
                                     A.HorizontalFlip(p = 0.5),
                                     #A.VerticalFlip(p = 0.5),
                                     A.ShiftScaleRotate(p = 0.01, rotate_limit = 30, shift_limit = 0.0625, scale_limit = 0.1, interpolation = cv2.INTER_LINEAR, border_mode = cv2.BORDER_CONSTANT),
                                     #A.RandomResizedCrop(p = 0.01, height = 512, width = 512, scale = [0.5, 1.]),
                                     A.ImageCompression(p = 0.01, quality_lower = 75),
                                    ],
                        min_area = 0., min_visibility = 0.,
                        batch_size = 0, epoch = 1, shuffle = False, prefetch = False, num_parallel_calls = None, cache = None, shuffle_size = None, prefetch_size = None):
    pipe = custom_pipe(albumentations, x_true, y_true, bbox_true, mask_true, dtype = None, tf_func = False,
                       transform = transform, min_area = min_area, min_visibility = min_visibility,
                       batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)
    return pipe
    
def key_map_pipe(x_true, y_true = None, bbox_true = None, mask_true = None, 
                 map = {"x_true":"x_true", "y_true":"y_true", "bbox_true":"bbox_true", "mask_true":"mask_true"},
                 batch_size = 0, epoch = 1, shuffle = False, prefetch = False, num_parallel_calls = None, cache = None, shuffle_size = None, prefetch_size = None):
    pipe = custom_pipe(key_map, x_true, y_true, bbox_true, mask_true, dtype = None, tf_func = True,
                       map = map,
                       batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)
    return pipe

def collect_pipe(x_true, y_true = None, bbox_true = None, mask_true = None, 
                 keys = ["x_true", "y_true", "bbox_true", "mask_true"],
                 batch_size = 0, epoch = 1, shuffle = False, prefetch = False, num_parallel_calls = None, cache = None, shuffle_size = None, prefetch_size = None):
    pipe = custom_pipe(collect, x_true, y_true, bbox_true, mask_true, dtype = None, tf_func = True,
                       keys = keys,
                       batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)
    return pipe