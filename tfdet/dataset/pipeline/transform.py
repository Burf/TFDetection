import functools

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf

from .util import pipe
from ..util import load_image
from ..pascal_voc import load_annotation
from tfdet.dataset import transform as T

def load(x_true, y_true = None, bbox_true = None, mask_true = None,
         load_func = load_image, anno_func = load_annotation, mask_func = None,
         dtype = None,
         batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
         cache = None, num_parallel_calls = None):
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.load, dtype = dtype, tf_func = False,
                load_func = load_func, anno_func = anno_func, mask_func = mask_func,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)
  
def preprocess(x_true, y_true = None, bbox_true = None, mask_true = None, 
               rescale = 1., mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375],
               label = None, one_hot = False, label_smoothing = 0.1,
               bbox_normalize = True, min_area = 0.,
               dtype = None,
               batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
               cache = None, num_parallel_calls = None):
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.preprocess, dtype = dtype, tf_func = False,
                rescale = rescale, mean = mean, std = std,
                label = label, one_hot = one_hot, label_smoothing = label_smoothing,
                bbox_normalize = bbox_normalize, min_area = min_area,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)
    
def resize(x_true, y_true = None, bbox_true = None, mask_true = None, 
           image_shape = None,
           dtype = None,
           batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
           cache = None, num_parallel_calls = None):
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.resize, dtype = dtype, tf_func = False,
                image_shape = image_shape,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def pad(x_true, y_true = None, bbox_true = None, mask_true = None, 
        image_shape = None, max_pad_size = 100, pad_val = 0, background = "bg", mode = "right",
        dtype = None,
        batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
        cache = None, num_parallel_calls = None):
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.pad, dtype = dtype, tf_func = False,
                image_shape = image_shape, max_pad_size = max_pad_size, pad_val = pad_val, background = background, mode = mode,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def crop(x_true, y_true = None, bbox_true = None, mask_true = None, 
         bbox = None, min_area = 0., min_visibility = 0.,
         dtype = None,
         batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
         cache = None, num_parallel_calls = None):
    """
    bbox = [x1, y1, x2, y2]
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.crop, dtype = dtype, tf_func = False,
                bbox = bbox, min_area = min_area, min_visibility = min_visibility,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def albumentations(x_true, y_true = None, bbox_true = None, mask_true = None,
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
                   dtype = None,
                   batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
                   cache = None, num_parallel_calls = None):
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.albumentations, dtype = dtype, tf_func = False,
                transform = transform, min_area = min_area, min_visibility = min_visibility,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def random_crop(x_true, y_true = None, bbox_true = None, mask_true = None,
                image_shape = None, min_area = 0., min_visibility = 0.,
                dtype = None,
                batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
                cache = None, num_parallel_calls = None):
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.random_crop, dtype = dtype, tf_func = False,
                image_shape = image_shape, min_area = min_area, min_visibility = min_visibility,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)
  
def mosaic(x_true, y_true = None, bbox_true = None, mask_true = None, 
           image_shape = None, alpha = 0.2, pad_val = 0, min_area = 0., min_visibility = 0., e = 1e-12, 
           dtype = None,
           pre_batch_size = 4, pre_shuffle = False, pre_shuffle_size = None,
           batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
           cache = None, num_parallel_calls = None):
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.mosaic, dtype = dtype, tf_func = False,
                image_shape = image_shape, alpha = alpha, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                pre_batch_size = pre_batch_size, pre_shuffle = pre_shuffle, pre_shuffle_size = pre_shuffle_size,
                cache = cache, num_parallel_calls = num_parallel_calls)
 
def cut_mix(x_true, y_true = None, bbox_true = None, mask_true = None,
            alpha = 1., min_area = 0., min_visibility = 0., e = 1e-12, 
            dtype = None,
            pre_batch_size = 2, pre_shuffle = False, pre_shuffle_size = None,
            batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
            cache = None, num_parallel_calls = None):
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.cut_mix, dtype = dtype, tf_func = False,
                alpha = alpha, min_area = min_area, min_visibility = min_visibility, e = e,
                pre_batch_size = pre_batch_size, pre_shuffle = pre_shuffle, pre_shuffle_size = pre_shuffle_size,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)
 
def cut_out(x_true, y_true = None, bbox_true = None, mask_true = None,
            alpha = 1., pad_val = 0, min_area = 0., min_visibility = 0., e = 1e-12,
            dtype = None,
            batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
            cache = None, num_parallel_calls = None):
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.cut_out, dtype = dtype, tf_func = False,
                alpha = alpha, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)
 
def mix_up(x_true, y_true = None, bbox_true = None, mask_true = None,
           alpha = 8.,
           dtype = None,
           pre_batch_size = 2, pre_shuffle = False, pre_shuffle_size = None,
           batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
           cache = None, num_parallel_calls = None):
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.mix_up, dtype = dtype, tf_func = False,
                alpha = alpha,
                pre_batch_size = pre_batch_size, pre_shuffle = pre_shuffle, pre_shuffle_size = pre_shuffle_size,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def random_mosaic(x_true, y_true = None, bbox_true = None, mask_true = None, 
                  p = 0.5,
                  image_shape = None, alpha = 0.2, pad_val = 0, min_area = 0., min_visibility = 0., e = 1e-12,
                  max_pad_size = 100, mode = "right", background = "bg",
                  dtype = None,
                  pre_batch_size = 16, pre_shuffle = False, pre_shuffle_size = None,
                  batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
                  cache = None, num_parallel_calls = None):
    func = functools.partial(T.mosaic, image_shape = image_shape, alpha = alpha, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e)
    random_func = functools.partial(T.random_apply, func, p = p, choice_size = 4, image_shape = image_shape, max_pad_size = max_pad_size, pad_val = pad_val, mode = mode, background = background)
    return pipe(x_true, y_true, bbox_true, mask_true, function = random_func, dtype = dtype, tf_func = False,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                pre_batch_size = pre_batch_size, pre_unbatch = True, pre_shuffle = pre_shuffle, pre_shuffle_size = pre_shuffle_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def random_cut_mix(x_true, y_true = None, bbox_true = None, mask_true = None, 
                   p = 0.5,
                   alpha = 1., min_area = 0., min_visibility = 0., e = 1e-12,
                   image_shape = None, max_pad_size = 100, pad_val = 0, mode = "right", background = "bg",
                   dtype = None,
                   pre_batch_size = 8, pre_shuffle = False, pre_shuffle_size = None,
                   batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
                   cache = None, num_parallel_calls = None):
    func = functools.partial(T.cut_mix, alpha = alpha, min_area = min_area, min_visibility = min_visibility, e = e)
    random_func = functools.partial(T.random_apply, func, p = p, choice_size = 2, image_shape = image_shape, max_pad_size = max_pad_size, pad_val = pad_val, mode = mode, background = background)
    return pipe(x_true, y_true, bbox_true, mask_true, function = random_func, dtype = dtype, tf_func = False,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                pre_batch_size = pre_batch_size, pre_unbatch = True, pre_shuffle = pre_shuffle, pre_shuffle_size = pre_shuffle_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def random_cut_out(x_true, y_true = None, bbox_true = None, mask_true = None, 
                   p = 0.5,
                   alpha = 1., pad_val = 0, min_area = 0., min_visibility = 0., e = 1e-12,
                   image_shape = None, max_pad_size = 100, mode = "right", background = "bg",
                   dtype = None,
                   batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
                   cache = None, num_parallel_calls = None):
    func = functools.partial(T.cut_out, alpha = alpha, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e)
    random_func = functools.partial(T.random_apply, func, p = p, choice_size = 1, image_shape = image_shape, max_pad_size = max_pad_size, pad_val = pad_val, mode = mode, background = background)
    return pipe(x_true, y_true, bbox_true, mask_true, function = random_func, dtype = dtype, tf_func = False,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                pre_batch_size = 1, pre_unbatch = True,
                cache = cache, num_parallel_calls = num_parallel_calls)

def random_mix_up(x_true, y_true = None, bbox_true = None, mask_true = None, 
                  p = 0.5,
                  alpha = 8.,
                  image_shape = None, max_pad_size = 100, pad_val = 0, mode = "right", background = "bg",
                  dtype = None,
                  pre_batch_size = 8, pre_shuffle = False, pre_shuffle_size = None,
                  batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
                  cache = None, num_parallel_calls = None):
    func = functools.partial(T.mix_up, alpha = alpha)
    random_func = functools.partial(T.random_apply, func, p = p, choice_size = 2, image_shape = image_shape, max_pad_size = max_pad_size, pad_val = pad_val, mode = mode, background = background)
    return pipe(x_true, y_true, bbox_true, mask_true, function = random_func, dtype = dtype, tf_func = False,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                pre_batch_size = pre_batch_size, pre_unbatch = True, pre_shuffle = pre_shuffle, pre_shuffle_size = pre_shuffle_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def key_map(x_true, y_true = None, bbox_true = None, mask_true = None, 
            map = {"x_true":"x_true", "y_true":"y_true", "bbox_true":"bbox_true", "mask_true":"mask_true"},
            dtype = None,
            batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
            cache = None, num_parallel_calls = None):
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.key_map, dtype = dtype, tf_func = True,
                map = map,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)

def collect(x_true, y_true = None, bbox_true = None, mask_true = None, 
            keys = ["x_true", "y_true", "bbox_true", "mask_true"],
            dtype = None,
            batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
            cache = None, num_parallel_calls = None):
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.collect, dtype = dtype, tf_func = True,
                keys = keys,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)

def cast(x_true, y_true = None, bbox_true = None, mask_true = None, 
         map = {"x_true":tf.float32, "y_true":tf.float32, "bbox_true":tf.float32, "mask_true":tf.float32},
         dtype = None,
         batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
         cache = None, num_parallel_calls = None):
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.cast, dtype = dtype, tf_func = True,
                map = map,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)

def args2dict(x_true, y_true = None, bbox_true = None, mask_true = None, 
              keys = ["x_true", "y_true", "bbox_true", "mask_true"],
              dtype = None,
              batch_size = 0, epoch = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
              cache = None, num_parallel_calls = None):
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.args2dict, dtype = dtype, tf_func = True,
                keys = keys,
                batch_size = batch_size, epoch = epoch, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)