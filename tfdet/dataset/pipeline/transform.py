import functools

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops

from .util import pipe, zip_pipe, concat_pipe, stack_pipe, dict_tf_func, convert_to_pickle
from ..util import load_image
from ..pascal_voc import load_annotation
from tfdet.dataset import transform as T

def load(x_true, y_true = None, bbox_true = None, mask_true = None,
         load_func = load_image, anno_func = load_annotation, mask_func = None,
         dtype = None,
         batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
         cache = False, num_parallel_calls = True):
    """
    x_true = [path, ...] or (N, H, W, C) or pipe
    y_true = [path, ...] or [annotation, ...]
    bbox_true = None or [annotation, ...]
    mask_true = [path, ...] or [annotation, ...]
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.load,
                load_func = load_func, anno_func = anno_func, mask_func = mask_func,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)
  
def normalize(x_true, y_true = None, bbox_true = None, mask_true = None, 
              rescale = 1., mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375],
              bbox_normalize = True,
              batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
              cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    x_true = ((x_true * rescale) - mean) / std (If variable is None, it does not apply.)
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype_info = {"x_true":tf.float32,
                  "y_true":None,
                  "bbox_true":tf.float32 if bbox_normalize else None,
                  "mask_true":None}
    if isinstance(pre_pipe.element_spec, dict):
        dtype = tuple([dtype_info[k] if dtype_info[k] is not None else v.dtype for k, v in pre_pipe.element_spec.items()])
    else:
        dtype_info = {i:dtype_info[k] for i, k in enumerate(["x_true", "y_true", "bbox_true", "mask_true"])}
        dtype = tuple([dtype_info[i] if dtype_info[i] is not None else v.dtype for i, v in enumerate(pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))])
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.normalize,
                rescale = rescale, mean = mean, std = std,
                bbox_normalize = bbox_normalize,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)
  
def unnormalize(x_true, y_true = None, bbox_true = None, mask_true = None, 
                rescale = 1., mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375],
                bbox_normalize = True,
                batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
                cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    x_true = ((x_true * std) + mean) / rescale (If variable is None, it does not apply.)
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype_info = {"x_true":tf.float32,
                  "y_true":None,
                  "bbox_true":tf.int32 if bbox_normalize else None,
                  "mask_true":None}
    if isinstance(pre_pipe.element_spec, dict):
        dtype = tuple([dtype_info[k] if dtype_info[k] is not None else v.dtype for k, v in pre_pipe.element_spec.items()])
    else:
        dtype_info = {i:dtype_info[k] for i, k in enumerate(["x_true", "y_true", "bbox_true", "mask_true"])}
        dtype = tuple([dtype_info[i] if dtype_info[i] is not None else v.dtype for i, v in enumerate(pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))])
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.unnormalize,
                rescale = rescale, mean = mean, std = std,
                bbox_normalize = bbox_normalize,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)

def filter_annotation(x_true, y_true = None, bbox_true = None, mask_true = None, 
                      label = None, min_scale = 2, min_instance_area = 1,
                      batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
                      cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    #Pad is removed.
    annotation = annotation[np.isin(y_true[..., 0], label)]
    annotation = annotation[min_scale[0] or min_scale <= bbox_height and min_scale[1] or min_scale <= bbox_width]
    annotation = annotation[min_instance_area <= instance_mask_area]
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.filter_annotation,
                label = label, min_scale = min_scale, min_instance_area = min_instance_area,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)
  
def label_encode(x_true, y_true = None, bbox_true = None, mask_true = None, 
                 label = None, one_hot = False, label_smoothing = 0.1,
                 batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
                 cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    label = ["background", ...]
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype_info = {"x_true":None,
                  "y_true":None,
                  "bbox_true":None,
                  "mask_true":tf.float32 if label is not None and one_hot and 0 < label_smoothing else None}
    if label is not None:
        dtype_info["y_true"] = tf.int32
        if one_hot and 0 < label_smoothing:
            dtype_info["y_true"] = tf.float32
    if isinstance(pre_pipe.element_spec, dict):
        dtype = tuple([dtype_info[k] if dtype_info[k] is not None else v.dtype for k, v in pre_pipe.element_spec.items()])
    else:
        dtype_info = {i:dtype_info[k] for i, k in enumerate(["x_true", "y_true", "bbox_true", "mask_true"])}
        dtype = tuple([dtype_info[i] if dtype_info[i] is not None else v.dtype for i, v in enumerate(pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))])
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.label_encode,
                label = label, one_hot = one_hot, label_smoothing = label_smoothing,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)
  
def label_decode(x_true, y_true = None, bbox_true = None, mask_true = None, 
                 label = None,
                 batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
                 cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    label = ["background", ...]
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype_info = {"x_true":None,
                  "y_true":tf.convert_to_tensor(label).dtype if label is not None else None,
                  "bbox_true":None,
                  "mask_true":None}
    if isinstance(pre_pipe.element_spec, dict):
        dtype = tuple([dtype_info[k] if dtype_info[k] is not None else v.dtype for k, v in pre_pipe.element_spec.items()])
    else:
        dtype_info = {i:dtype_info[k] for i, k in enumerate(["x_true", "y_true", "bbox_true", "mask_true"])}
        dtype = tuple([dtype_info[i] if dtype_info[i] is not None else v.dtype for i, v in enumerate(pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))])
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.label_decode,
                label = label,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)
                
def compose(x_true, y_true = None, bbox_true = None, mask_true = None,
            transform = [], dtype = None,
            batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
            cache = False, num_parallel_calls = True,
            **kwargs):
    
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    if dtype is None:
        pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
        dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
        dtype = [spec.dtype for spec in dtype]
        dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    func = functools.partial(T.compose, transform = transform, **kwargs)
    return pipe(x_true, y_true, bbox_true, mask_true, function = func,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)
    
def resize(x_true, y_true = None, bbox_true = None, mask_true = None, 
           image_shape = None, keep_ratio = True, method = cv2.INTER_LINEAR, mode = "value",
           batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
           cache = False, num_parallel_calls = True):
    
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    image_shape = [h, w] or [[h, w], ...] for value / jitter mode, [min_scale, max_scale] or [[min_scale, max_scale], ...] for range mode
    mode = ("value", "range", "jitter")
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.resize,
                image_shape = image_shape, keep_ratio = keep_ratio, method = method, mode = mode,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)

def pad(x_true, y_true = None, bbox_true = None, mask_true = None, 
        image_shape = None, shape_divisor = None, max_pad_size = 100, pad_val = 114, mode = "both", background = "background",
        batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
        cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    mode = ("left", "right", "both", "random")
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.pad,
                image_shape = image_shape, shape_divisor = shape_divisor, max_pad_size = max_pad_size, pad_val = pad_val, mode = mode, background = background,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)

def trim(x_true, y_true = None, bbox_true = None, mask_true = None, 
         image_shape = None, pad_val = 114, mode = "both", min_area = 0., min_visibility = 0., e = 1e-12, decimal = 4,
         batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
         cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    #Pad is removed.
    pad_val = np.round(x_true, decimal)'s pad_val
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.trim,
                image_shape = image_shape, pad_val = pad_val, mode = mode, min_area = min_area, min_visibility = min_visibility, e = e, decimal = decimal,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)

def crop(x_true, y_true = None, bbox_true = None, mask_true = None, 
         bbox = None, min_area = 0., min_visibility = 0., e = 1e-12,
         batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
         cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    #Pad is removed.
    bbox = [x1, y1, x2, y2]
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.crop,
                bbox = bbox, min_area = min_area, min_visibility = min_visibility, e = e,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)

def flip(x_true, y_true = None, bbox_true = None, mask_true = None,
         mode = "horizontal",
         batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
         cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    mode = ("horizontal", "vertical")
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.flip,
                mode = mode,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)

def random_crop(x_true, y_true = None, bbox_true = None, mask_true = None,
                image_shape = None, min_area = 0., min_visibility = 0., e = 1e-12,
                batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
                cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C)
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)

    #Pad is removed.
    #If image_shape is shape or ratio, apply random_crop.
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.random_crop,
                image_shape = image_shape, min_area = min_area, min_visibility = min_visibility, e = e,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)

def random_flip(x_true, y_true = None, bbox_true = None, mask_true = None, 
                p = 0.5, mode = "horizontal",
                batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
                cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)

    mode = ("horizontal", "vertical")
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.random_flip,
                p = p, mode = mode,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)

def multi_scale_flip(x_true, y_true = None, bbox_true = None, mask_true = None,
                     image_shape = None, keep_ratio = True, flip_mode = "horizontal", method = cv2.INTER_LINEAR, resize_mode = "value",
                     shape_divisor = None, max_pad_size = 100, pad_val = 114, pad_mode = "both", background = "background",
                     batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
                     cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    image_shape = [h, w](single apply) or [[h, w], ...](multi apply) for value / jitter mode, [min_scale, max_scale](single apply) or [[min_scale, max_scale], ...](multi apply) for range mode
    flip_mode = ("horizontal", "vertical", None)(single apply) or [mode, ...](multi apply)
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    pre_pipe = pipe(pre_pipe, shuffle = shuffle, repeat = repeat)
    
    aug_pipes = []
    for shape in ([image_shape] if np.ndim(image_shape) < 2 else image_shape):
        resize_pipe = pre_pipe
        if shape is not None:
            resize_pipe = resize(pre_pipe, image_shape = shape, keep_ratio = keep_ratio, method = method, mode = resize_mode, num_parallel_calls = num_parallel_calls)
            resize_pipe = pad(resize_pipe, image_shape = shape, shape_divisor = shape_divisor, max_pad_size = max_pad_size, pad_val = pad_val, mode = pad_mode, background = background, num_parallel_calls = num_parallel_calls)
            aug_pipes.append(resize_pipe.batch(batch_size) if 0 < batch_size else resize_pipe)
        for m in ([flip_mode] if np.ndim(flip_mode) < 1 else flip_mode):
            if m is not None:
                flip_pipe = flip(resize_pipe, mode = m, num_parallel_calls = num_parallel_calls)
                aug_pipes.append(flip_pipe.batch(batch_size) if 0 < batch_size else flip_pipe)
    
    concat_pipe = pre_pipe
    if 0 < len(aug_pipes):
        concat_pipe = aug_pipes[0]
        for p in aug_pipes[1:]:
            concat_pipe = concat_pipe.concatenate(p)
    elif 0 < batch_size:
        concat_pipe = concat_pipe.batch(batch_size)
    return pipe(concat_pipe, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls)

def yolo_hsv(x_true, y_true = None, bbox_true = None, mask_true = None, 
             h = 0.015, s = 0.7, v = 0.4,
             batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
             cache = False, num_parallel_calls = True):
    """
    https://github.com/WongKinYiu/yolov7/blob/main/utils/datasets.py
    
    x_true = (N, H, W, C) or pipe #RGB, np.uint8
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.yolo_hsv,
                h = h, s = s, v = v,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)

def random_perspective(x_true, y_true = None, bbox_true = None, mask_true = None, 
                       image_shape = None, perspective = 0., rotate = 0., translate = 0.2, scale = 0.9, shear = 0., pad_val = 114, min_area = 0., min_visibility = 0., e = 1e-12,
                       batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
                       cache = False, num_parallel_calls = True):
    """
    https://github.com/WongKinYiu/yolov7/blob/main/utils/datasets.py
    
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    #Pad is removed.
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.random_perspective,
                image_shape = image_shape, perspective = perspective, rotate = rotate, translate = translate, scale = scale, shear = shear, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)
  
def mosaic(x_true, y_true = None, bbox_true = None, mask_true = None, 
           sample_x_true = None, sample_y_true = None, sample_bbox_true = None, sample_mask_true = None,
           p = 0.5,
           image_shape = None, alpha = 0.25, pad_val = 114, min_area = 0., min_visibility = 0., e = 1e-12,
           sample_cache = False, sample_shuffle = True,
           batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
           cache = False, num_parallel_calls = True):
    """
    https://github.com/WongKinYiu/yolov7/blob/main/utils/datasets.py
    
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    usage > tfdet.dataset.pipeline.mosaic(tr_pipe.cache("./train"), sample_x_true = sample_pipe.cache("./sample"))
    
    #If image_shape is None, the result is (N, 2 * H, 2 * W, C).
    """
    if isinstance(x_true, tf.data.Dataset) and isinstance(y_true, tf.data.Dataset) and sample_x_true is None:
        sample_x_true = y_true
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    
    sample_pipe = (sample_x_true if isinstance(sample_x_true, tf.data.Dataset) else pipe(sample_x_true, sample_y_true, sample_bbox_true, sample_mask_true)) if sample_x_true is not None else pre_pipe
    if sample_cache and not isinstance(sample_pipe, dataset_ops.CacheDataset):
        sample_pipe = pipe(sample_pipe, cache = sample_cache)
    pre_pipe, sample_pipe = pre_pipe.map(convert_to_pickle), sample_pipe.map(convert_to_pickle)
    args_pipe = concat_pipe(pre_pipe.batch(1), (sample_pipe.shuffle(3 * 10) if sample_shuffle else sample_pipe).repeat().batch(3), axis = 0)
    
    func = functools.partial(T.mosaic, image_shape = image_shape, alpha = alpha, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e)
    def fail_func(x_true, y_true = None, bbox_true = None, mask_true = None, image_shape = image_shape):
        x_true = x_true[0]
        y_true = y_true[0] if y_true is not None else None
        bbox_true = bbox_true[0] if bbox_true is not None else None
        mask_true = mask_true[0] if mask_true is not None else None
        return T.pad(x_true, y_true, bbox_true, mask_true, image_shape = image_shape if image_shape is not None else 2, max_pad_size = 0)
    random_func = functools.partial(T.random_apply, function = [func, fail_func], p = p)
    return pipe(args_pipe, function = random_func,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)

def mosaic9(x_true, y_true = None, bbox_true = None, mask_true = None, 
            sample_x_true = None, sample_y_true = None, sample_bbox_true = None, sample_mask_true = None,
            p = 0.5,
            image_shape = None, pad_val = 114, min_area = 0., min_visibility = 0., e = 1e-12,
            sample_cache = False, sample_shuffle = True,
            batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
            cache = False, num_parallel_calls = True):
    """
    https://github.com/WongKinYiu/yolov7/blob/main/utils/datasets.py
    
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    usage > tfdet.dataset.pipeline.mosaic9(tr_pipe.cache("./train"), sample_x_true = sample_pipe.cache("./sample"))
    
    #If image_shape is None, the result is (N, 2 * H, 2 * W, C).
    """
    if isinstance(x_true, tf.data.Dataset) and isinstance(y_true, tf.data.Dataset) and sample_x_true is None:
        sample_x_true = y_true
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    
    sample_pipe = (sample_x_true if isinstance(sample_x_true, tf.data.Dataset) else pipe(sample_x_true, sample_y_true, sample_bbox_true, sample_mask_true)) if sample_x_true is not None else pre_pipe
    if sample_cache and not isinstance(sample_pipe, dataset_ops.CacheDataset):
        sample_pipe = pipe(sample_pipe, cache = sample_cache)
    pre_pipe, sample_pipe = pre_pipe.map(convert_to_pickle), sample_pipe.map(convert_to_pickle)
    args_pipe = concat_pipe(pre_pipe.batch(1), (sample_pipe.shuffle(8 * 10) if sample_shuffle else sample_pipe).repeat().batch(8), axis = 0)
    
    func = functools.partial(T.mosaic9, image_shape = image_shape, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e)
    def fail_func(x_true, y_true = None, bbox_true = None, mask_true = None, image_shape = image_shape):
        x_true = x_true[0]
        y_true = y_true[0] if y_true is not None else None
        bbox_true = bbox_true[0] if bbox_true is not None else None
        mask_true = mask_true[0] if mask_true is not None else None
        return T.pad(x_true, y_true, bbox_true, mask_true, image_shape = image_shape if image_shape is not None else 2, max_pad_size = 0)
    random_func = functools.partial(T.random_apply, function = [func, fail_func], p = p)
    return pipe(args_pipe, function = random_func,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)

def cut_mix(x_true, y_true = None, bbox_true = None, mask_true = None, 
            sample_x_true = None, sample_y_true = None, sample_bbox_true = None, sample_mask_true = None,
            p = 0.5,
            alpha = 1., min_area = 0., min_visibility = 0., e = 1e-12,
            batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
            cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    
    sample_pipe = (sample_x_true if isinstance(sample_x_true, tf.data.Dataset) else pipe(sample_x_true, sample_y_true, sample_bbox_true, sample_mask_true)) if sample_x_true is not None else pre_pipe
    args_pipe = concat_pipe(pre_pipe.batch(1), sample_pipe.batch(1), axis = 0)
    
    func = functools.partial(T.cut_mix, alpha = alpha, min_area = min_area, min_visibility = min_visibility, e = e)
    random_func = functools.partial(T.random_apply, function = func, p = p, reduce = True)
    return pipe(args_pipe, function = random_func,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)

def cut_out(x_true, y_true = None, bbox_true = None, mask_true = None, 
            p = 0.5,
            alpha = 1., pad_val = 114, min_area = 0., min_visibility = 0., e = 1e-12,
            batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
            cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    
    func = functools.partial(T.cut_out, alpha = alpha, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e)
    random_func = functools.partial(T.random_apply, function = func, p = p, reduce = True)
    return pipe(x_true, y_true, bbox_true, mask_true, function = random_func,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)

def mix_up(x_true, y_true = None, bbox_true = None, mask_true = None, 
           sample_x_true = None, sample_y_true = None, sample_bbox_true = None, sample_mask_true = None,
           p = 0.15,
           alpha = 8.,
           batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
           cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    #dtype = tuple([tf.float32] + list(dtype[1:])) if isinstance(dtype, tuple) else tf.float32
    
    sample_pipe = (sample_x_true if isinstance(sample_x_true, tf.data.Dataset) else pipe(sample_x_true, sample_y_true, sample_bbox_true, sample_mask_true)) if sample_x_true is not None else pre_pipe
    args_pipe = concat_pipe(pre_pipe.batch(1), sample_pipe.batch(1), axis = 0)
    
    func = functools.partial(T.mix_up, alpha = alpha)
    random_func = functools.partial(T.random_apply, function = func, p = p, reduce = True)
    return pipe(args_pipe, function = random_func,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)

def copy_paste(x_true, y_true = None, bbox_true = None, mask_true = None, 
               sample_x_true = None, sample_y_true = None, sample_bbox_true = None, sample_mask_true = None,
               p = 0.15,
               max_paste_count = 100, scale_range = [0.0625, 0.75], clip_object = True, replace = True, random_count = True, label = None,
               min_scale = 2, min_instance_area = 1, iou_threshold = 0.3,
               copy_min_scale = 2, copy_min_instance_area = 1, copy_iou_threshold = 0.3,
               p_flip = 0.5, pad_val = 114, method = cv2.INTER_LINEAR,
               min_area = 0., min_visibility = 0., e = 1e-12,
               sample_size = 4, sample_cache = False, sample_shuffle = True,
               batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
               cache = False, num_parallel_calls = True):
    """
    https://arxiv.org/abs/2012.07177
    
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    usage > tfdet.dataset.pipeline.copy_paste(tr_pipe.cache("./train"), sample_x_true = sample_pipe.cache("./sample"))
    
    #First image is Background image.
    #Paste object condition : min_scale[0] or min_scale <= paste_object_height and min_scale[1] or min_scale <= paste_object_width
    #Paste mask condition : min_instance_area <= paste_instance_mask_area
    scale = np.random.beta(1, 1.4) * np.abs(scale_range[1] - scale_range[0]) + np.min(scale_range)
    clip_object = Don't crop object
    replace = np.random.choice's replace
    random_count = change max_paste_count from 0 to max_paste_count
    label = copy target label
    iou_threshold = iou_threshold or [copy_iou_threshold, paste_iou_threshold]
    """
    if isinstance(x_true, tf.data.Dataset) and isinstance(y_true, tf.data.Dataset) and sample_x_true is None:
        sample_x_true = y_true
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    
    sample_pipe = (sample_x_true if isinstance(sample_x_true, tf.data.Dataset) else pipe(sample_x_true, sample_y_true, sample_bbox_true, sample_mask_true)) if sample_x_true is not None else pre_pipe
    if sample_cache and not isinstance(sample_pipe, dataset_ops.CacheDataset):
        sample_pipe = pipe(sample_pipe, cache = sample_cache)
    pre_pipe, sample_pipe = pre_pipe.map(convert_to_pickle), sample_pipe.map(convert_to_pickle)
    args_pipe = concat_pipe(pre_pipe.batch(1), (sample_pipe.shuffle(max(sample_size, 1) * 10) if sample_shuffle else sample_pipe).repeat().batch(max(sample_size, 1)), axis = 0)
        
    func = functools.partial(T.copy_paste, max_paste_count = max_paste_count, scale_range = scale_range, clip_object = clip_object, replace = replace, random_count = random_count, label = label, min_scale = min_scale, min_instance_area = min_instance_area, iou_threshold = iou_threshold, copy_min_scale = copy_min_scale, copy_min_instance_area = copy_min_instance_area, copy_iou_threshold = copy_iou_threshold, p_flip = p_flip, pad_val = pad_val, method = method, min_area = min_area, min_visibility = min_visibility, e = e)
    def fail_func(x_true, y_true = None, bbox_true = None, mask_true = None):
        result = [v[0] for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
        result = result[0] if len(result) == 1 else tuple(result)
        return result
    random_func = functools.partial(T.random_apply, function = [func, fail_func], p = p)
    return pipe(args_pipe, function = random_func,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)

def remove_background(x_true, y_true = None, bbox_true = None, mask_true = None, 
                      pad_val = 114,
                      batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
                      cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.remove_background,
                pad_val = pad_val,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)

def yolo_augmentation(x_true, y_true = None, bbox_true = None, mask_true = None,
                      sample_x_true = None, sample_y_true = None, sample_bbox_true = None, sample_mask_true = None,
                      image_shape = None, pad_val = 114,
                      perspective = 0., rotate = 0., translate = 0.2, scale = 0.9, shear = 0.,
                      h = 0.015, s = 0.7, v = 0.4,
                      max_paste_count = 20, scale_range = [0.0625, 0.75], clip_object = True, replace = True, random_count = False, label = None,
                      min_scale = 2, min_instance_area = 1, iou_threshold = 0.3, copy_min_scale = 2, copy_min_instance_area = 1, copy_iou_threshold = 0.3, p_copy_paste_flip = 0.5, method = cv2.INTER_LINEAR,
                      p_mosaic = 1., p_mix_up = 0.15, p_copy_paste = 0., p_flip = 0.5, p_mosaic9 = 0.8,
                      min_area = 0., min_visibility = 0., e = 1e-12,
                      sample_size = 8 + 9 + 4, sample_cache = False, sample_shuffle = True,
                      batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
                      cache = False, num_parallel_calls = True):
    """
    https://github.com/WongKinYiu/yolov7/blob/main/utils/datasets.py
    
    x_true = (N, H, W, C)
    y_true(without bbox_true) = (N, n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    usage > tfdet.dataset.pipeline.yolo_augmentation(tr_pipe.cache("./train"), sample_x_true = sample_pipe.cache("./sample"))
    
    #(mosaic + random_perspective > mix_up(with sample mosaic + random_perspective)) or (pad + random_perspective) > yolo_hsv > copy_paste(optional) > random_flip
    #First image is Background image.
    #If image_shape is shape or ratio, apply random_crop.
    """
    if isinstance(x_true, tf.data.Dataset) and isinstance(y_true, tf.data.Dataset) and sample_x_true is None:
        sample_x_true = y_true
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    #dtype = tuple([tf.float32] + list(dtype[1:])) if isinstance(dtype, tuple) else tf.float32
    
    sample_pipe = (sample_x_true if isinstance(sample_x_true, tf.data.Dataset) else pipe(sample_x_true, sample_y_true, sample_bbox_true, sample_mask_true)) if sample_x_true is not None else pre_pipe
    if sample_cache and not isinstance(sample_pipe, dataset_ops.CacheDataset):
        sample_pipe = pipe(sample_pipe, cache = sample_cache)
    pre_pipe, sample_pipe = pre_pipe.map(convert_to_pickle), sample_pipe.map(convert_to_pickle)
    args_pipe = concat_pipe(pre_pipe.batch(1), (sample_pipe.shuffle(max(sample_size, 1) * 10) if sample_shuffle else sample_pipe).repeat().batch(max(sample_size, 1)), axis = 0)
    
    func = functools.partial(T.yolo_augmentation, image_shape = image_shape, pad_val = pad_val,
                             perspective = perspective, rotate = rotate, translate = translate, scale = scale, shear = shear,
                             h = h, s = s, v = v,
                             max_paste_count = max_paste_count, scale_range = scale_range, clip_object = clip_object, replace = replace, random_count = random_count, label = label,
                             min_scale = min_scale, min_instance_area = min_instance_area, iou_threshold = iou_threshold, copy_min_scale = copy_min_scale, copy_min_instance_area = copy_min_instance_area, copy_iou_threshold = copy_iou_threshold, p_copy_paste_flip = p_copy_paste_flip, method = method,
                             p_mosaic = p_mosaic, p_mix_up = p_mix_up, p_copy_paste = p_copy_paste, p_flip = p_flip, p_mosaic9 = p_mosaic9,
                             min_area = min_area, min_visibility = min_visibility, e = e)
    return pipe(args_pipe, function = func,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)

def mmdet_augmentation(x_true, y_true = None, bbox_true = None, mask_true = None,
                       image_shape = [1333, 800], keep_ratio = True, crop_shape = None, p_flip = 0.5,
                       flip_mode = "horizontal", method = cv2.INTER_LINEAR, resize_mode = "jitter",
                       shape_divisor = 32, max_pad_size = 100, pad_val = 114, pad_mode = "both", background = "background",
                       min_area = 0., min_visibility = 0., e = 1e-12,
                       batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
                       cache = False, num_parallel_calls = True):
    """
    https://github.com/open-mmlab/mmdetection/blob/master/configs/_base_/datasets/coco_detection.py
    
    x_true = (N, H, W, C)
    y_true(without bbox_true) = (N, n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    #random_resize > random_crop(optional) > random_flip > pad(by shape_divisor)
    #If crop_shape is shape or ratio, apply random_crop.
    #Pad is removed.(by random crop)
    """
    pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
    dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
    dtype = [spec.dtype for spec in dtype]
    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.mmdet_augmentation,
                image_shape = image_shape, keep_ratio = keep_ratio, p_flip = p_flip, crop_shape = crop_shape,
                flip_mode = flip_mode, method = method, resize_mode = resize_mode,
                shape_divisor = shape_divisor, max_pad_size = max_pad_size, pad_val = pad_val, pad_mode = pad_mode, background = background,
                min_area = min_area, min_visibility = min_visibility, e = e,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                cache = cache, num_parallel_calls = num_parallel_calls,
                tf_func = False, dtype = dtype)

try:
    import albumentations as A
    
    def albumentations(x_true, y_true = None, bbox_true = None, mask_true = None,
                       transform = [A.CLAHE(p = 0.1, clip_limit = 4., tile_grid_size = (8, 8)),
                                    A.RandomBrightnessContrast(p = 0.1, brightness_limit = 0.2, contrast_limit = 0.2),
                                    A.RandomGamma(p = 0.1, gamma_limit = [80, 120]),
                                    A.Blur(p = 0.1),
                                    A.MedianBlur(p = 0.1),
                                    A.ToGray(p = 0.1),
                                    A.RGBShift(p = 0.1, r_shift_limit = 10, g_shift_limit = 10, b_shift_limit = 10),
                                    A.HueSaturationValue(p = 0.1, hue_shift_limit = 10, sat_shift_limit = 40, val_shift_limit = 50),
                                    A.ChannelShuffle(p = 0.1),
                                    #A.ShiftScaleRotate(p = 0.1, rotate_limit = 30, shift_limit = 0.0625, scale_limit = 0.1, interpolation = cv2.INTER_LINEAR, border_mode = cv2.BORDER_CONSTANT),
                                    #A.RandomResizedCrop(p = 0.1, height = 512, width = 512, scale = [0.8, 1.0], ratio = [0.9, 1.11]),
                                    A.ImageCompression(p = 0.1, quality_lower = 75),
                                   ],
                       min_area = 0., min_visibility = 0.,
                       batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
                       cache = False, num_parallel_calls = True):
        """
        x_true = (N, H, W, C) or pipe
        y_true(without bbox_true) = (N, 1 or n_class)
        y_true(with bbox_true) = (N, P, 1 or n_class)
        bbox_true = (N, P, 4)
        mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
        mask_true(semantic mask_true) = (N, H, W, 1 or n_class)

        #Pad is removed.
        """
        pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
        dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
        dtype = [spec.dtype for spec in dtype]
        dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
        return pipe(x_true, y_true, bbox_true, mask_true, function = T.albumentations,
                    transform = transform, min_area = min_area, min_visibility = min_visibility,
                    batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                    cache = cache, num_parallel_calls = num_parallel_calls,
                    tf_func = False, dtype = dtype)
    
    
    def weak_augmentation(x_true, y_true = None, bbox_true = None, mask_true = None,
                          crop_shape = None, 
                          transform = [A.CLAHE(p = 0.1, clip_limit = 4., tile_grid_size = (8, 8)),
                                       A.RandomBrightnessContrast(p = 0.1, brightness_limit = 0.2, contrast_limit = 0.2),
                                       A.RandomGamma(p = 0.1, gamma_limit = [80, 120]),
                                       A.Blur(p = 0.1),
                                       A.MedianBlur(p = 0.1),
                                       A.ToGray(p = 0.1),
                                       A.RGBShift(p = 0.1, r_shift_limit = 10, g_shift_limit = 10, b_shift_limit = 10),
                                       A.HueSaturationValue(p = 0.1, hue_shift_limit = 10, sat_shift_limit = 40, val_shift_limit = 50),
                                       A.ChannelShuffle(p = 0.1),
                                       #A.ShiftScaleRotate(p = 0.1, rotate_limit = 30, shift_limit = 0.0625, scale_limit = 0.1, interpolation = cv2.INTER_LINEAR, border_mode = cv2.BORDER_CONSTANT),
                                       #A.RandomResizedCrop(p = 0.1, height = 512, width = 512, scale = [0.8, 1.0], ratio = [0.9, 1.11]),
                                       A.ImageCompression(p = 0.1, quality_lower = 75),
                                      ],
                          p_flip = 0.5, flip_mode = "horizontal",
                          min_area = 0., min_visibility = 0., e = 1e-12,
                          batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
                          cache = False, num_parallel_calls = True):
        """
        x_true = (N, H, W, C) or pipe
        y_true(without bbox_true) = (N, 1 or n_class)
        y_true(with bbox_true) = (N, P, 1 or n_class)
        bbox_true = (N, P, 4)
        mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
        mask_true(semantic mask_true) = (N, H, W, 1 or n_class)

        #albumentations > random_flip > random_crop(optional)
        #Pad is removed.
        #If crop_shape is shape or ratio, apply random_crop.
        """
        pre_pipe = x_true if isinstance(x_true, tf.data.Dataset) else pipe(x_true, y_true, bbox_true, mask_true)
        dtype = list(pre_pipe.element_spec.values()) if isinstance(pre_pipe.element_spec, dict) else (pre_pipe.element_spec if isinstance(pre_pipe.element_spec, tuple) else (pre_pipe.element_spec,))
        dtype = [spec.dtype for spec in dtype]
        dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
        return pipe(x_true, y_true, bbox_true, mask_true, function = T.weak_augmentation,
                    crop_shape = crop_shape, transform = transform, p_flip = p_flip, flip_mode = flip_mode, min_area = min_area, min_visibility = min_visibility, e = e,
                    batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                    cache = cache, num_parallel_calls = num_parallel_calls,
                    tf_func = False, dtype = dtype)
except:
    pass

def key_map(x_true, y_true = None, bbox_true = None, mask_true = None, 
            map = {"x_true":"x_true", "y_true":"y_true", "bbox_true":"bbox_true", "mask_true":"mask_true"},
            batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
            cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.key_map,
                map = map,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache,
                tf_func = True)

def collect(x_true, y_true = None, bbox_true = None, mask_true = None, 
            keys = ["x_true", "y_true", "bbox_true", "mask_true"],
            batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
            cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.collect,
                keys = keys,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache,
                tf_func = True)

def cast(x_true, y_true = None, bbox_true = None, mask_true = None, 
         map = {"x_true":tf.float32, "y_true":tf.float32, "bbox_true":tf.float32, "mask_true":tf.float32},
         batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
         cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.cast,
                map = map,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache,
                tf_func = True)

def args2dict(x_true, y_true = None, bbox_true = None, mask_true = None, 
              keys = ["x_true", "y_true", "bbox_true", "mask_true"],
              batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
              cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.args2dict,
                keys = keys,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache,
                tf_func = True)

def dict2args(x_true, y_true = None, bbox_true = None, mask_true = None, 
              keys = None,
              batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
              cache = False, num_parallel_calls = True):
    """
    x_true = (N, H, W, C) or pipe
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    return pipe(x_true, y_true, bbox_true, mask_true, function = T.dict2args,
                keys = keys,
                batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, num_parallel_calls = num_parallel_calls, cache = cache,
                tf_func = True)