import random

import cv2
import numpy as np

from tfdet.core.util import dict_function, to_categorical
from ..util import load_image, trim_bbox, pad as pad_numpy
from ..pascal_voc import load_annotation, load_instance

def load(x_true, y_true = None, bbox_true = None, mask_true = None, load_func = load_image, anno_func = load_annotation, mask_func = load_instance):
    """
    x_true = path or (H, W, C)
    y_true = path or annotation
    bbox_true = None or annotation
    mask_true = path or annotation
    """
    if callable(load_func):
        x_true = load_func(x_true)
        if isinstance(x_true, tuple):
            out = list(x_true)
            x_true = out.pop(0)
            if 0 < len(out):
                y_true = out.pop(0)
            if 0 < len(out):
                bbox_true = out.pop(0)
            if 0 < len(out):
                mask_true = out.pop(0)
    
    if y_true is not None:
        if callable(anno_func):
            y_true = anno_func(y_true, bbox_true)
        if isinstance(y_true, tuple):
            out = list(y_true)
            y_true = out.pop(0)
            if 0 < len(out):
                bbox_true = out.pop(0)
            if 0 < len(out):
                mask_true = out.pop(0)
    
    if mask_true is not None:
        if callable(mask_func):
            mask_true = mask_func(mask_true)
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def normalize(x_true, y_true = None, bbox_true = None, mask_true = None,
              rescale = 1., mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375],
              bbox_normalize = True):
    """
    x_true = (H, W, C)
    y_true(without bbox_true) = (1 or n_class)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    
    x_true = ((x_true * rescale) - mean) / std (If variable is None, it does not apply.)
    """
    if rescale is not None:
        x_true = np.multiply(x_true, rescale)
    if mean is not None:
        x_true = np.subtract(x_true, mean)
    if std is not None:
        x_true = np.divide(x_true, std)
    if bbox_true is not None:
        h, w = np.shape(x_true)[-3:-1]
        window = [w, h, w, h] if np.any(np.greater_equal(bbox_true, 2)) else 1
        if bbox_normalize:
            bbox_true = np.divide(bbox_true, window)
            window = 1
        bbox_true = np.clip(bbox_true, 0, window)
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def unnormalize(x_true, y_true = None, bbox_true = None, mask_true = None,
                rescale = 1., mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375],
                bbox_normalize = True):
    """
    x_true = (H, W, C)
    y_true(without bbox_true) = (1 or n_class)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    
    x_true = ((x_true * std) + mean) / rescale (If variable is None, it does not apply.)
    """
    if std is not None:
        x_true = np.multiply(x_true, std)
    if mean is not None:
        x_true = np.add(x_true, mean)
    if rescale is not None:
        x_true = np.divide(x_true, rescale)
    if bbox_true is not None:
        h, w = np.shape(x_true)[-3:-1]
        window = [w, h, w, h] if np.any(np.greater_equal(bbox_true, 2)) else 1
        if bbox_normalize and not np.any(np.greater_equal(bbox_true, 2)):
            window = [w, h, w, h]
            bbox_true = np.round(np.multiply(bbox_true, window)).astype(int)
        bbox_true = np.clip(bbox_true, 0, window)
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def filter_annotation(x_true, y_true = None, bbox_true = None, mask_true = None,
                      label = None, min_scale = 2, min_instance_area = 1):
    """
    x_true = (H, W, C)
    y_true(without bbox_true) = (1 or n_class)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    
    #The pad will be removed.
    annotation = annotation[np.isin(y_true[..., 0], label)]
    annotation = annotation[min_scale[0] or min_scale <= bbox_height and min_scale[1] or min_scale <= bbox_width]
    annotation = annotation[min_instance_area <= instance_mask_area]
    """
    if y_true is not None and label is not None:
        y_true = np.array(y_true)
        flag = np.isin(y_true[..., 0], label if 0 < np.ndim(label) else [label])
        y_true = y_true[flag]
        if bbox_true is not None:
            bbox_true = np.array(bbox_true)[flag]
        if mask_true is not None and 3 < np.ndim(mask_true):
            mask_true = np.array(mask_true)[flag]
    if bbox_true is not None:
        h, w = np.shape(x_true)[:2]
        bbox_true = np.array(bbox_true)
        min_scale = [min_scale, min_scale] if np.ndim(min_scale) == 0 else min_scale
        if 2 <= np.max(min_scale):
            if not np.any(np.greater_equal(bbox_true, 2)):
                min_scale = np.divide(min_scale, [h, w])
        else:
            if np.any(np.greater_equal(bbox_true, 2)):
                min_scale = np.multiply(min_scale, [h, w])
        flag = np.logical_and(min_scale[0] <= (bbox_true[..., 3] - bbox_true[..., 1]), min_scale[1] <= (bbox_true[..., 2] - bbox_true[..., 0]))
        bbox_true = bbox_true[flag]
        if y_true is not None and 1 < np.ndim(y_true):
            y_true = np.array(y_true)[flag]
        if mask_true is not None and 3 < np.ndim(mask_true):
            mask_true = np.array(mask_true)[flag]
    if mask_true is not None and 3 < np.ndim(mask_true):
        area = np.sum(np.greater(mask_true, 0.5), axis = (1, 2, 3))
        flag = min_instance_area <= area
        mask_true = mask_true[flag]
        if y_true is not None:
            y_true = np.array(y_true)[flag]
        if bbox_true is not None:
            bbox_true = np.array(bbox_true)[flag]
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result
    
def label_encode(x_true, y_true = None, bbox_true = None, mask_true = None, 
                 label = None, one_hot = False, label_smoothing = 0.1):
    """
    x_true = (H, W, C)
    y_true(without bbox_true) = (1 or n_class)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    
    label = ["background", ...]
    """
    if y_true is not None and label is not None:
        if 0 < len(y_true):
            label_convert = {k:v for v, k in enumerate(label if 0 < np.ndim(label) else [label])} if not isinstance(label, dict) else label
            if 1 < np.ndim(y_true):
                y_true = np.array([[label_convert[l[0]]] if l[0] in label else l for l in y_true])
            else:
                y_true = label_convert[y_true] if y_true in label else y_true
        if one_hot:
            y_true = np.array(to_categorical(y_true, len(label), label_smoothing))
    if (mask_true is not None and np.ndim(mask_true) < 4) and (label is not None and one_hot):
        mask_true = np.array(to_categorical(mask_true, len(label), label_smoothing))
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def label_decode(x_true, y_true = None, bbox_true = None, mask_true = None, label = None):
    """
    x_true = (H, W, C)
    y_true(without bbox_true) = (1 or n_class)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    
    label = ["background", ...]
    """
    if y_true is not None and label is not None:
        if 0 < len(y_true):
            if 0 < np.ndim(y_true) and np.shape(y_true)[-1] != 1:
                y_true = np.expand_dims(np.argmax(y_true, axis = -1), axis = -1)
            y_true = np.array(label)[np.array(y_true, dtype = np.int32)]
    if mask_true is not None and np.ndim(mask_true) < 4:
        if 0 < len(mask_true) and 2 < np.ndim(mask_true) and np.shape(mask_true)[-1] != 1:
            mask_true = np.expand_dims(np.argmax(mask_true, axis = -1), axis = -1)
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result
    
def compose(x_true, y_true = None, bbox_true = None, mask_true = None,
            transform = [],
            **kwargs):
    transform = [transform] if callable(transform) else transform
    if 0 < len(transform):
        func = dict_function(transform, keys = ["x_true", "y_true", "bbox_true", "mask_true"])
        result = func(x_true, y_true, bbox_true, mask_true, **kwargs)
    else:
        result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
        result = result[0] if len(result) == 1 else tuple(result)
    return result
    
def resize(x_true, y_true = None, bbox_true = None, mask_true = None, image_shape = None, keep_ratio = True, method = cv2.INTER_LINEAR):
    """
    x_true = (H, W, C)
    y_true(without bbox_true) = (1 or n_class)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    
    image_shape = [h, w] or [[h, w], ...](random choice)
    """
    if image_shape is not None:
        if 1 < np.ndim(image_shape):
            if 1 < choice_size:
                #image_shape = random.choice(image_shape)
                image_shape = image_shape[np.random.choice(np.arange(len(image_shape)))] #for numpy seed
        elif np.ndim(image_shape) == 0:
            image_shape = np.round(np.multiply(np.shape(x_true)[:2], image_shape)).astype(int)
        target_size = tuple(image_shape[:2])
        size = np.shape(x_true)[:2]
        if keep_ratio:
            scale = min(max(target_size) / max(size), min(target_size) / min(size))
            target_size = tuple((np.multiply(size, scale) + 0.5).astype(int))
        if target_size != size:
            target_size = target_size[::-1]
            x_true = cv2.resize(x_true, target_size, interpolation = method)
            if bbox_true is not None and np.any(np.greater_equal(bbox_true, 2)):
                bbox_true = np.multiply(np.divide(bbox_true, np.tile(size[::-1], 2)), np.tile(target_size, 2))
                bbox_true = np.round(bbox_true).astype(int)
            if mask_true is not None:
                if 3 < np.ndim(mask_true):
                    mask_true = np.expand_dims([cv2.resize(m, target_size, interpolation = method) for m in mask_true], axis = -1)
                else:
                    mask_true = cv2.resize(mask_true, target_size, interpolation = method)
                    mask_true = np.expand_dims(mask_true, axis = -1) if np.ndim(mask_true) == 2 else mask_true
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def pad(x_true, y_true = None, bbox_true = None, mask_true = None, image_shape = None, shape_divisor = None, max_pad_size = 100, pad_val = 114, mode = "both", background = "background"):
    """
    x_true = (H, W, C)
    y_true(without bbox_true) = (1 or n_class)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    
    mode = ("left", "right", "both", "random")
    """
    if mode not in ("left", "right", "both", "random"):
        raise ValueError("unknown mode '{0}'".format(mode))
    
    if image_shape is not None and np.ndim(image_shape) == 0:
        image_shape = np.round(np.multiply(np.shape(x_true)[:2], image_shape)).astype(int)
    elif image_shape is None and shape_divisor is not None:
        image_shape = (np.ceil(np.divide(np.shape(x_true)[:2], shape_divisor)) * shape_divisor).astype(int)
    h, w = np.shape(x_true)[:2]
    new_h, new_w = image_shape[:2] if image_shape is not None else [h, w]
    
    l = r = [0, 0]
    p = [max(new_h - h, 0), max(new_w - w, 0)]
    if mode == "left":
        l = p
    elif mode == "right":
        r = p
    elif mode == "both":
        l = np.divide(p, 2).astype(int)
        r = np.subtract(p, l)
    elif mode == "random":
        l = np.random.randint(0, np.add(p, 1))
        r = np.subtract(p, l)
    if 0 < np.max([l, r]):
        x_true = pad_numpy(x_true, [[l[0], r[0]], [l[1], r[1]], [0, 0]], val = pad_val)
    if bbox_true is not None:
        indices = np.where(np.max(0 < bbox_true, axis = -1, keepdims = True) != 0)[0]
        bbox_true = np.array(bbox_true)[indices]
        if y_true is not None:
            y_true = np.array(y_true)[indices]
        if mask_true is not None and 3 < np.ndim(mask_true):
            mask_true = np.array(mask_true)[indices]
        
        if 0 < np.max(l):
            if np.any(np.greater_equal(bbox_true, 2)):
                bbox_true = np.add(bbox_true, np.tile(l[::-1], 2))
            else:
                bbox_true = np.multiply(bbox_true, np.tile(np.divide([w, h], [new_w, new_h]), 2))
                bbox_true = np.add(bbox_true, np.tile(np.divide(l[::-1], [new_w, new_h]), 2))
        #bbox_true = bbox_true[:max_pad_size]
        bbox_true =  pad_numpy(bbox_true, [[0, max(max_pad_size - len(bbox_true), 0)], [0, 0]])
    if y_true is not None and 1 < np.ndim(y_true):
        val = 0 if np.issubdtype((np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true).dtype, np.number) else background
        #y_true = y_true[:max_pad_size]
        y_true = pad_numpy(y_true, [[0, max(max_pad_size - len(y_true), 0)], [0, 0]], val = val)
    if mask_true is not None:
        if 3 < np.ndim(mask_true):
            #mask_true = mask_true[:max_pad_size]
            mask_true = pad_numpy(mask_true, [[0, max(max_pad_size - len(mask_true), 0)], [l[0], r[0]], [l[1], r[1]], [0, 0]])
        elif 0 < np.max([l, r]):
            mask_true = pad_numpy(mask_true, [[l[0], r[0]], [l[1], r[1]], [0, 0]])
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def trim(x_true, y_true = None, bbox_true = None, mask_true = None, image_shape = None, pad_val = 114, mode = "both", min_area = 0., min_visibility = 0., e = 1e-12, decimal = 4):
    """
    x_true = (H, W, C)
    y_true(without bbox_true) = (1 or n_class)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    
    #The pad will be removed.
    pad_val = np.round(x_true, decimal)'s pad_val
    """
    if image_shape is not None and np.ndim(image_shape) == 0:
        image_shape = np.round(np.multiply(np.shape(x_true)[:2], image_shape)).astype(int)
    h, w = np.shape(x_true)[:2]
    bbox = trim_bbox(x_true, image_shape = image_shape, pad_val = pad_val, mode = mode, decimal = decimal)
    if True: #not np.all(np.equal(bbox, [0, 0, w, h])):
        return crop(x_true, y_true, bbox_true, mask_true, bbox = bbox, min_area = min_area, min_visibility = min_visibility, e = e)
    else:
        result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
        result = result[0] if len(result) == 1 else tuple(result)
        return result

def crop(x_true, y_true = None, bbox_true = None, mask_true = None, bbox = None, min_area = 0., min_visibility = 0., e = 1e-12):
    """
    x_true = (H, W, C)
    y_true(without bbox_true) = (1 or n_class)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    
    #The pad will be removed.
    bbox = [x1, y1, x2, y2]
    """
    h, w = np.shape(x_true)[:2]
    bbox = [0, 0, w, h] if bbox is None else bbox
    bbox = np.round(np.multiply(np.clip(bbox, 0, 1), [w, h, w, h])).astype(int) if np.max(bbox) < 2 else np.clip(bbox, 0, [w, h, w, h]).astype(int)
        
    x_true = np.array(x_true[bbox[1]:bbox[3], bbox[0]:bbox[2]])
    if mask_true is not None and bbox_true is None:
        mask_true = np.array(mask_true)[..., bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    if bbox_true is not None:
        new_h, new_w = np.shape(x_true)[:2]
        bbox_unnorm = np.any(np.greater_equal(bbox_true, 2))
        unnorm_bbox = np.round(np.multiply(bbox_true, [w, h, w, h])).astype(int) if not bbox_unnorm else np.array(bbox_true)

        #new_bbox = np.maximum(unnorm_bbox, [bbox[2], bbox[3], bbox[2], bbox[3]])
        new_bbox = np.subtract(unnorm_bbox, [bbox[0], bbox[1], bbox[0], bbox[1]])
        new_bbox = np.clip(new_bbox, 0, [new_w, new_h, new_w, new_h])

        area = (unnorm_bbox[..., 2] - unnorm_bbox[..., 0]) * (unnorm_bbox[..., 3] - unnorm_bbox[..., 1])
        new_area = (new_bbox[..., 2] - new_bbox[..., 0]) * (new_bbox[..., 3] - new_bbox[..., 1])

        flag = np.logical_and(min_area <= (new_area / (new_w * new_h)), min_visibility < (new_area / (area + e)))
        bbox_true = new_bbox[flag]
        bbox_true = np.divide(bbox_true, [new_w, new_h, new_w, new_h]) if not bbox_unnorm else bbox_true
        if y_true is not None and 1 < np.ndim(y_true):
            y_true = np.array(y_true)[flag]
        if mask_true is not None:
            mask_true = np.array(mask_true)
            if 3 < np.ndim(mask_true):
                mask_true = mask_true[flag]
            mask_true = mask_true[..., bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def flip(x_true, y_true = None, bbox_true = None, mask_true = None, mode = "horizontal"):
    """
    x_true = (H, W, C)
    y_true(without bbox_true) = (1 or n_class)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    
    mode = ("horizontal", "vertical")
    """
    if mode not in ("horizontal", "vertical", "both"):
        raise ValueError("unknown mode '{0}'".format(mode))
        
    shape = np.shape(x_true)[:2]
    func = np.fliplr if mode == "horizontal" else np.flipud
    x_true = func(x_true)
    if bbox_true is not None:
        bbox_true = np.array(bbox_true)
        if not np.any(np.greater_equal(bbox_true, 2)): #bbox_norm
            shape = [1, 1]
        x1, y1, x2, y2 = np.split(bbox_true, 4, axis = -1)
        w, h = x2 - x1, y2 - y1
        x, y = x1 + (0.5 * w), y1 + (0.5 * h)
        if mode == "horizontal":
            x = shape[1] - x
        else:
            y = shape[0] - y
        x1, y1 = x - (0.5 * w), y - (0.5 * h)
        x2, y2 = x1 + w, y1 + h
        bbox_true = np.concatenate([x1, y1, x2, y2], axis = -1)
    if mask_true is not None:
        mask_true = np.array([func(m) for m in mask_true]) if 3 < np.ndim(mask_true) else func(mask_true)
    
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def random_apply(x_true, y_true = None, bbox_true = None, mask_true = None, function = None, p = 0.5, reduce = False, **kwargs):
    """
    x_true = (H, W, C) or (N, H, W, C)
    y_true(without bbox_true) = (n_class) or (N, n_class)
    y_true(with bbox_true) = (P, 1 or n_class) or (N, P, 1 or n_class)
    bbox_true = (P, 4) or (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1) or (N, P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class) or (N, H, W, 1 or n_class)
    """
    fail_function = None
    if 0 < np.ndim(function) and 1 < len(function):
        fail_function = function[1]
        function = function[0]
    
    if callable(function) and np.random.random() < p:
        result = function(x_true, y_true, bbox_true, mask_true, **kwargs)
    elif callable(fail_function):
        result = fail_function(x_true, y_true, bbox_true, mask_true, **kwargs)
    else:
        if reduce and 3 < np.ndim(x_true):
            x_true = x_true[0]
            y_true = y_true[0] if y_true is not None else None
            bbox_true = bbox_true[0] if bbox_true is not None else None
            mask_true = mask_true[0] if mask_true is not None else None
        result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
        result = result[0] if len(result) == 1 else tuple(result)
    return result

def random_shuffle_apply(x_true, y_true = None, bbox_true = None, mask_true = None, function = None, p = 0.5, sample_size = 1, replace = True,
                         image_shape = None, shape_divisor = None, max_pad_size = 100, pad_val = 114, mode = "both", background = "background", **kwargs):
    """
    x_true = (N, H, W, C)
    y_true(without bbox_true) = (N, n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    fail_function = None
    if 0 < np.ndim(function) and 1 < len(function):
        fail_function = function[1]
        function = function[0]
    
    n_batch = len(x_true)
    indices = np.arange(n_batch)
    x_trues = []
    y_trues = []
    bbox_trues = []
    mask_trues = []
    for index in range(n_batch):
        if callable(function) and np.random.random() < p:
            if 1 < sample_size:
                #index = [index] + random.choices(indices, k = min(n_batch - 1, sample_size - 1) if not replace else (sample_size - 1))
                index = [index] + np.random.choice(indices, min(n_batch - 1, sample_size - 1) if not replace else (sample_size - 1), replace = replace).tolist() #for numpy seed
            out = function(x_true[index], 
                           y_true[index] if y_true is not None else None,
                           bbox_true[index] if bbox_true is not None else None, 
                           mask_true[index] if mask_true is not None else None, **kwargs)
            if not isinstance(out, (tuple, list)):
                out = (out,)
        elif callable(fail_function):
            if 1 < sample_size:
                #index = [index] + random.choices(indices, k = min(n_batch - 1, sample_size - 1) if not replace else (sample_size - 1))
                index = [index] + np.random.choice(indices, min(n_batch - 1, sample_size - 1) if not replace else (sample_size - 1), replace = replace).tolist() #for numpy seed
            out = fail_function(x_true[index], 
                                y_true[index] if y_true is not None else None,
                                bbox_true[index] if bbox_true is not None else None, 
                                mask_true[index] if mask_true is not None else None, **kwargs)
            if not isinstance(out, (tuple, list)):
                out = (out,)
        else:
            out = [arg[index] for arg in [x_true, y_true, bbox_true, mask_true] if arg is not None]
        args = {k:v for k, v in zip(["x_true", "y_true", "bbox_true", "mask_true"], out)}
        out = pad(**args, image_shape = image_shape, shape_divisor = shape_divisor, max_pad_size = max_pad_size, pad_val = pad_val, mode = mode, background = background)
        if not isinstance(out, (tuple, list)):
            out = (out,)
        out = list(out)
        x_trues.append(out.pop(0))
        if y_true is not None and 0 < len(out):
            y_trues.append(out.pop(0))
        if bbox_true is not None and 0 < len(out):
            bbox_trues.append(out.pop(0))
        if mask_true is not None and 0 < len(out):
            mask_trues.append(out.pop(0))
    try:
        x_true = np.stack(x_trues, axis = 0)
        y_true = np.stack(y_trues, axis = 0) if y_true is not None else None
        bbox_true = np.stack(bbox_trues, axis = 0) if bbox_true is not None else None
        mask_true = np.stack(mask_trues, axis = 0) if mask_true is not None else None
    except:
        raise ValueError("all input arrays must have the same shape : please check 'image_shape' or 'shape_divisor' or 'max_pad_size'")
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result