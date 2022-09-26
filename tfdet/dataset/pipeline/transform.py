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
         batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
         cache = None, num_parallel_calls = None):
    """
    x_true = [path, ...] or (N, H, W, C) or pipe
    y_true = [path, ...] or [annotation, ...]
    bbox_true = None or [annotation, ...]
    mask_true = [path, ...] or [annotation, ...]
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.load, dtype = dtype, tf_func = False,
                load_func = load_func, anno_func = anno_func, mask_func = mask_func,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)
  
def normalize(x_true, y_true = None, bbox_true = None, mask_true = None, 
              rescale = 1., mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375],
              bbox_normalize = True,
              dtype = None,
              batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
              cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    x_true = ((x_true * rescale) - mean) / std (If variable is None, it does not apply.)
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.normalize, dtype = dtype, tf_func = False,
                rescale = rescale, mean = mean, std = std,
                bbox_normalize = bbox_normalize,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)
  
def unnormalize(x_true, y_true = None, bbox_true = None, mask_true = None, 
                rescale = 1., mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375],
                bbox_normalize = True,
                dtype = None,
                batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
                cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    x_true = ((x_true * std) + mean) / rescale (If variable is None, it does not apply.)
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.unnormalize, dtype = dtype, tf_func = False,
                rescale = rescale, mean = mean, std = std,
                bbox_normalize = bbox_normalize,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def filter_annotation(x_true, y_true = None, bbox_true = None, mask_true = None, 
                      label = None, min_scale = 2, min_instance_area = 1,
                      dtype = None,
                      batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
                      cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    #The pad will be removed.
    annotation = annotation[np.isin(y_true[..., 0], label)]
    annotation = annotation[min_scale[0] or min_scale <= bbox_height and min_scale[1] or min_scale <= bbox_width]
    annotation = annotation[min_instance_area <= instance_mask_area]
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.filter_annotation, dtype = dtype, tf_func = False,
                label = label, min_scale = min_scale, min_instance_area = min_instance_area,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)
  
def label_encode(x_true, y_true = None, bbox_true = None, mask_true = None, 
                 label = None, one_hot = False, label_smoothing = 0.1,
                 dtype = None,
                 batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
                 cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.label_encode, dtype = dtype, tf_func = False,
                label = label, one_hot = one_hot, label_smoothing = label_smoothing,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)
  
def label_decode(x_true, y_true = None, bbox_true = None, mask_true = None, 
                 label = None,
                 dtype = None,
                 batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
                 cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.label_decode, dtype = dtype, tf_func = False,
                label = label,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)
    
def resize(x_true, y_true = None, bbox_true = None, mask_true = None, 
           image_shape = None, keep_ratio = True,
           dtype = None,
           batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
           cache = None, num_parallel_calls = None):
    
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    image_shape = [h, w] or [[h, w], ...](random choice)
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.resize, dtype = dtype, tf_func = False,
                image_shape = image_shape, keep_ratio = keep_ratio,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def pad(x_true, y_true = None, bbox_true = None, mask_true = None, 
        image_shape = None, shape_divisor = None, max_pad_size = 100, pad_val = 114, mode = "both", background = "bg",
        dtype = None,
        batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
        cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    mode = ("left", "right", "both", "random")
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.pad, dtype = dtype, tf_func = False,
                image_shape = image_shape, shape_divisor = shape_divisor, max_pad_size = max_pad_size, pad_val = pad_val, mode = mode, background = background,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def trim(x_true, y_true = None, bbox_true = None, mask_true = None, 
         image_shape = None, pad_val = 114, mode = "both", min_area = 0., min_visibility = 0., decimal = 4,
         dtype = None,
         batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
         cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    #The pad will be removed.
    pad_val = np.round(x_true, decimal)'s pad_val
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.trim, dtype = dtype, tf_func = False,
                image_shape = image_shape, pad_val = pad_val, mode = mode, min_area = min_area, min_visibility = min_visibility, decimal = decimal,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def crop(x_true, y_true = None, bbox_true = None, mask_true = None, 
         bbox = None, min_area = 0., min_visibility = 0.,
         dtype = None,
         batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
         cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    #The pad will be removed.
    bbox = [x1, y1, x2, y2]
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.crop, dtype = dtype, tf_func = False,
                bbox = bbox, min_area = min_area, min_visibility = min_visibility,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
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
                                A.ShiftScaleRotate(p = 0.01, rotate_limit = 30, shift_limit = 0.0625, scale_limit = 0.1, interpolation = cv2.INTER_LINEAR, border_mode = cv2.BORDER_CONSTANT),
                                #A.RandomResizedCrop(p = 0.01, height = 512, width = 512, scale = [0.5, 1.]),
                                A.ImageCompression(p = 0.01, quality_lower = 75),
                               ],
                   min_area = 0., min_visibility = 0.,
                   dtype = None,
                   batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
                   cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    #The pad will be removed.
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.albumentations, dtype = dtype, tf_func = False,
                transform = transform, min_area = min_area, min_visibility = min_visibility,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def random_crop(x_true, y_true = None, bbox_true = None, mask_true = None,
                image_shape = None, min_area = 0., min_visibility = 0.,
                dtype = None,
                batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
                cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C)
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    #The pad will be removed.
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.random_crop, dtype = dtype, tf_func = False,
                image_shape = image_shape, min_area = min_area, min_visibility = min_visibility,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def random_flip(x_true, y_true = None, bbox_true = None, mask_true = None, 
                p = 0.5, mode = "horizontal",
                dtype = None,
                batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
                cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    #The pad will be removed.
    mode = ("horizontal", "vertical", "both")
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.random_flip, dtype = dtype, tf_func = False,
                p = p, mode = mode,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def multi_scale_flip(x_true, y_true = None, bbox_true = None, mask_true = None,
                     image_shape = None, keep_ratio = True, flip = True, mode = "horizontal",
                     dtype = None,
                     batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
                     cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    image_shape = [h, w](single apply) or [[h, w], ...](multi apply)
    mode = ("horizontal", "vertical", "both")(single apply) or [mode, ...](multi apply)
    """
    pre_pipe = pipe(x_true, y_true, bbox_true, mask_true)
    if dtype is None:
        dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else pre_pipe.element_spec
        dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    
    aug_pipes = []
    for shape in ([image_shape] if np.ndim(image_shape) < 2 else image_shape):
        resize_pipe = resize(pre_pipe, image_shape = shape, keep_ratio = keep_ratio, dtype = dtype) if shape is not None else pre_pipe
        aug_pipes.append(resize_pipe)
        if flip:
            for m in ([mode] if np.ndim(mode) < 1 else mode):
                flip_pipe = random_flip(resize_pipe, p = 1., mode = m, dtype = dtype)
                aug_pipes.append(flip_pipe)
    
    concat_pipe = pre_pipe
    if 0 < len(aug_pipes):
        concat_pipe = aug_pipes[0]
        for p in aug_pipes[1:]:
            concat_pipe = concat_pipe.concatenate(p)
    return pipe(concat_pipe,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def yolo_hsv(x_true, y_true = None, bbox_true = None, mask_true = None, 
             h = 0.015, s = 0.7, v = 0.4,
             dtype = None,
             batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
             cache = None, num_parallel_calls = None):
    """
    https://github.com/WongKinYiu/yolov7/blob/main/utils/datasets.py
    
    x_true = (N, H, W, C) or pipe #RGB, np.uint8
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.yolo_hsv, dtype = dtype, tf_func = False,
                h = h, s = s, v = v,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def random_perspective(x_true, y_true = None, bbox_true = None, mask_true = None, 
                       image_shape = None, perspective = 0., rotate = 0., translate = 0.2, scale = 0.9, shear = 0., pad_val = 114, min_area = 0., min_visibility = 0., e = 1e-12,
                       dtype = None,
                       batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
                       cache = None, num_parallel_calls = None):
    """
    https://github.com/WongKinYiu/yolov7/blob/main/utils/datasets.py
    
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    #The pad will be removed.
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.random_perspective, dtype = dtype, tf_func = False,
                image_shape = image_shape, perspective = perspective, rotate = rotate, translate = translate, scale = scale, shear = shear, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)
  
def mosaic(x_true, y_true = None, bbox_true = None, mask_true = None, 
           p = 0.5,
           image_shape = None, alpha = 0.25, pad_val = 114, min_area = 0., min_visibility = 0., e = 1e-12,
           shape_divisor = None, max_pad_size = 100, mode = "both", background = "bg",
           dtype = None,
           pre_batch_size = 16, pre_shuffle = False, pre_shuffle_size = None, choice_size = 4,
           batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
           cache = None, num_parallel_calls = None):
    """
    https://github.com/WongKinYiu/yolov7/blob/main/utils/datasets.py
    
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
        
    #The pad will be removed.
    #If image_shape is None, the result is (N, 2 * H, 2 * W, C).
    """
    func = functools.partial(T.mosaic, image_shape = image_shape, alpha = alpha, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e)
    random_func = functools.partial(T.random_apply, func, p = p, choice_size = choice_size, image_shape = image_shape, shape_divisor = shape_divisor, max_pad_size = max_pad_size, pad_val = pad_val, mode = mode, background = background)
    return pipe(x_true, y_true, bbox_true, mask_true, function = random_func, dtype = dtype, tf_func = False,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                pre_batch_size = pre_batch_size, pre_unbatch = True, pre_shuffle = pre_shuffle, pre_shuffle_size = pre_shuffle_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def mosaic9(x_true, y_true = None, bbox_true = None, mask_true = None, 
            p = 0.5,
            image_shape = None, pad_val = 114, min_area = 0., min_visibility = 0.,
            shape_divisor = None, max_pad_size = 100, mode = "both", background = "bg",
            dtype = None,
            pre_batch_size = 36, pre_shuffle = False, pre_shuffle_size = None, choice_size = 9, 
            batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
            cache = None, num_parallel_calls = None):
    """
    https://github.com/WongKinYiu/yolov7/blob/main/utils/datasets.py
    
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
        
    #The pad will be removed.
    #If image_shape is None, the result is (N, 2 * H, 2 * W, C).
    """
    func = functools.partial(T.mosaic9, image_shape = image_shape, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility)
    random_func = functools.partial(T.random_apply, func, p = p, choice_size = choice_size, image_shape = image_shape, shape_divisor = shape_divisor, max_pad_size = max_pad_size, pad_val = pad_val, mode = mode, background = background)
    return pipe(x_true, y_true, bbox_true, mask_true, function = random_func, dtype = dtype, tf_func = False,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                pre_batch_size = pre_batch_size, pre_unbatch = True, pre_shuffle = pre_shuffle, pre_shuffle_size = pre_shuffle_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def cut_mix(x_true, y_true = None, bbox_true = None, mask_true = None, 
            p = 0.5,
            alpha = 1., min_area = 0., min_visibility = 0., e = 1e-12,
            image_shape = None, shape_divisor = None, max_pad_size = 100, pad_val = 114, mode = "both", background = "bg",
            dtype = None,
            pre_batch_size = 4, pre_shuffle = False, pre_shuffle_size = None, choice_size = 2,
            batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
            cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
        
    #The pad will be removed.
    """
    func = functools.partial(T.cut_mix, alpha = alpha, min_area = min_area, min_visibility = min_visibility, e = e)
    random_func = functools.partial(T.random_apply, func, p = p, choice_size = choice_size, image_shape = image_shape, shape_divisor = shape_divisor, max_pad_size = max_pad_size, pad_val = pad_val, mode = mode, background = background)
    return pipe(x_true, y_true, bbox_true, mask_true, function = random_func, dtype = dtype, tf_func = False,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                pre_batch_size = pre_batch_size, pre_unbatch = True, pre_shuffle = pre_shuffle, pre_shuffle_size = pre_shuffle_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def cut_out(x_true, y_true = None, bbox_true = None, mask_true = None, 
            p = 0.5,
            alpha = 1., pad_val = 114, min_area = 0., min_visibility = 0., e = 1e-12,
            image_shape = None, shape_divisor = None, max_pad_size = 100, mode = "both", background = "bg",
            dtype = None,
            batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
            cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
        
    #The pad will be removed.
    """
    func = functools.partial(T.cut_out, alpha = alpha, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e)
    random_func = functools.partial(T.random_apply, func, p = p, choice_size = 1, image_shape = image_shape, shape_divisor = shape_divisor, max_pad_size = max_pad_size, pad_val = pad_val, mode = mode, background = background)
    return pipe(x_true, y_true, bbox_true, mask_true, function = random_func, dtype = dtype, tf_func = False,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                pre_batch_size = 1, pre_unbatch = True,
                cache = cache, num_parallel_calls = num_parallel_calls)

def mix_up(x_true, y_true = None, bbox_true = None, mask_true = None, 
           p = 0.15,
           alpha = 8.,
           image_shape = None, shape_divisor = None, max_pad_size = 100, pad_val = 114, mode = "both", background = "bg",
           dtype = None,
           pre_batch_size = 4, pre_shuffle = False, pre_shuffle_size = None, choice_size = 2,
           batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
           cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    #The pad will be removed.
    """
    func = functools.partial(T.mix_up, alpha = alpha)
    random_func = functools.partial(T.random_apply, func, p = p, choice_size = choice_size, image_shape = image_shape, shape_divisor = shape_divisor, max_pad_size = max_pad_size, pad_val = pad_val, mode = mode, background = background)
    return pipe(x_true, y_true, bbox_true, mask_true, function = random_func, dtype = dtype, tf_func = False,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                pre_batch_size = pre_batch_size, pre_unbatch = True, pre_shuffle = pre_shuffle, pre_shuffle_size = pre_shuffle_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def copy_paste(x_true, y_true = None, bbox_true = None, mask_true = None, 
               p = 0.15,
               max_paste_count = 20, scale_range = [0.03125, 0.75], clip_object = True, replace = True,
               min_scale = 2, min_instance_area = 1, iou_threshold = 0.3, p_flip = 0.5, method = cv2.INTER_LINEAR,
               min_area = 0., min_visibility = 0., e = 1e-12,
               image_shape = None, shape_divisor = None, max_pad_size = 100, pad_val = 114, mode = "both", background = "bg",
               dtype = None,
               pre_batch_size = 16, pre_shuffle = False, pre_shuffle_size = None, choice_size = 4,
               batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
               cache = None, num_parallel_calls = None):
    """
    https://arxiv.org/abs/2012.07177
    
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    #The pad will be removed.
    #First image is Background image.
    #Paste object condition : min_scale[0] or min_scale <= paste_object_height and min_scale[1] or min_scale <= paste_object_width
    #Paste mask condition : min_instance_area <= paste_instance_mask_area
    scale = np.random.beta(1, 1.3) * np.abs(scale_range[1] - scale_range[0]) + np.min(scale_range)
    clip_object = Don't crop object
    replace = np.random.choice's replace
    """
    func = functools.partial(T.copy_paste, max_paste_count = max_paste_count, scale_range = scale_range, clip_object = clip_object, replace = replace, min_scale = min_scale, min_instance_area = min_instance_area, iou_threshold = iou_threshold, p_flip = p_flip, method = method, min_area = min_area, min_visibility = min_visibility, e = e)
    random_func = functools.partial(T.random_apply, func, p = p, choice_size = choice_size, image_shape = image_shape, shape_divisor = shape_divisor, max_pad_size = max_pad_size, pad_val = pad_val, mode = mode, background = background)
    return pipe(x_true, y_true, bbox_true, mask_true, function = random_func, dtype = dtype, tf_func = False,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                pre_batch_size = pre_batch_size, pre_unbatch = True, pre_shuffle = pre_shuffle, pre_shuffle_size = pre_shuffle_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def remove_background(x_true, y_true = None, bbox_true = None, mask_true = None, 
                      pad_val = 114,
                      dtype = None,
                      batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
                      cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.remove_background, dtype = dtype, tf_func = False,
                pad_val = pad_val,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                cache = cache, num_parallel_calls = num_parallel_calls)

def key_map(x_true, y_true = None, bbox_true = None, mask_true = None, 
            map = {"x_true":"x_true", "y_true":"y_true", "bbox_true":"bbox_true", "mask_true":"mask_true"},
            dtype = None,
            batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
            cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.key_map, dtype = dtype, tf_func = True,
                map = map,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)

def collect(x_true, y_true = None, bbox_true = None, mask_true = None, 
            keys = ["x_true", "y_true", "bbox_true", "mask_true"],
            dtype = None,
            batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
            cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.collect, dtype = dtype, tf_func = True,
                keys = keys,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)

def cast(x_true, y_true = None, bbox_true = None, mask_true = None, 
         map = {"x_true":tf.float32, "y_true":tf.float32, "bbox_true":tf.float32, "mask_true":tf.float32},
         dtype = None,
         batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
         cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.cast, dtype = dtype, tf_func = True,
                map = map,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)

def args2dict(x_true, y_true = None, bbox_true = None, mask_true = None, 
              keys = ["x_true", "y_true", "bbox_true", "mask_true"],
              dtype = None,
              batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
              cache = None, num_parallel_calls = None):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.args2dict, dtype = dtype, tf_func = True,
                keys = keys,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache, shuffle_size = shuffle_size, prefetch_size = prefetch_size)