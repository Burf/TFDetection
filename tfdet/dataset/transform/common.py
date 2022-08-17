import albumentations as A
import cv2
import numpy as np

from tfdet.core.util import to_categorical
from .augment import albumentations
from ..util import load_image, load_pascal_voc

def load(x_true, y_true = None, bbox_true = None, mask_true = None, load_func = load_image, anno_func = load_pascal_voc, mask_func = None):
    if callable(load_func):
        x_true = load_func(x_true)
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
    
def preprocess(x_true, y_true = None, bbox_true = None, mask_true = None, 
               rescale = 1., mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375],
               label = None, one_hot = False, label_smoothing = 0.1,
               bbox_normalize = True, min_area = 0.):
    """
    x_true = (H, W, C)
    y_true(without bbox_true) = (1 or n_class,)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    """
    if bbox_true is not None:
        h, w = np.shape(x_true)[:2]
        if bbox_normalize and np.any(np.greater(bbox_true, 1)):
            bbox_true = np.divide(bbox_true, [w, h, w, h])
        if np.any(np.greater(bbox_true, 1)):
            bbox_true = np.clip(bbox_true, 0., [w, h, w, h])
        else:
            bbox_true = np.clip(bbox_true, 0., 1.)
        if 0 < min_area:
            area = (bbox_true[..., 3] - bbox_true[..., 1]) * (bbox_true[..., 2] - bbox_true[..., 0])
            flag = min_area <= (area if np.max(area) <= 1 else area / (h * w))
            bbox_true = bbox_true[flag]
    if rescale is not None:
        x_true = np.multiply(x_true, rescale)
    if mean is not None:
        x_true = np.subtract(x_true, mean)
    if std is not None:
        x_true = np.divide(x_true, std)
    if y_true is not None and label is not None:
        if 0 < len(y_true):
            label_convert = {k:v for v, k in enumerate(label)}
            if 1 < np.ndim(y_true):
                y_true = np.array([[label_convert[l[0]]] if l[0] in label else l for l in y_true])
            else:
                y_true = label_convert[y_true] if y_true in label else y_true
        if one_hot:
            y_true = to_categorical(y_true, len(label), label_smoothing)
        if 0 < min_area and bbox_true is not None and 1 < np.ndim(y_true):
            y_true = y_true[flag]
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result
    
def resize(x_true, y_true = None, bbox_true = None, mask_true = None, image_shape = None, method = cv2.INTER_LINEAR):
    if image_shape is not None:
        target_size = tuple(image_shape[:2])
        size = np.shape(x_true)[:2]
        if target_size != size:
            target_size = target_size[::-1]
            x_true = cv2.resize(x_true, target_size, interpolation = method)
            if bbox_true is not None and np.any(np.greater(bbox_true, 1)):
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

def pad(x_true, y_true = None, bbox_true = None, mask_true = None, image_shape = None, max_pad_size = 100, pad_val = 0, mode = "right", background = "bg"):
    """
    x_true = (H, W, C)
    y_true(without bbox_true) = (1 or n_class,)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    
    mode = ("left", "right", "both", "random")
    """
    if mode not in ("left", "right", "both", "random"):
        raise ValueError("unknown mode '{0}'".format(mode))
    
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
    x_true = np.pad(x_true, [[l[0], r[0]], [l[1], r[1]], [0, 0]])
    if y_true is not None and 1 < np.ndim(y_true):
        val = background if 0 < len(y_true) and isinstance(y_true[0][0], str) else 0
        #y_true = y_true[:max_pad_size]
        y_true = np.pad(y_true, [[0, max(max_pad_size - len(y_true), 0)], [0, 0]], constant_values = val)
    if bbox_true is not None:
        if np.any(np.greater(bbox_true, 1)):
            bbox_true = np.add(bbox_true, np.tile(l[::-1], 2))
        else:
            bbox_true = np.multiply(bbox_true, np.tile(np.divide([w, h], [new_w, new_h]), 2))
            bbox_true = np.add(bbox_true, np.tile(np.divide(l[::-1], [new_w, new_h]), 2))
        #bbox_true = bbox_true[:max_pad_size]
        bbox_true =  np.pad(bbox_true, [[0, max(max_pad_size - len(bbox_true), 0)], [0, 0]])
    if mask_true is not None:
        if 3 < np.ndim(mask_true):
            #mask_true = mask_true[:max_pad_size]
            mask_true = np.pad(mask_true, [[0, max(max_pad_size - len(mask_true), 0)], [l[0], r[0]], [l[1], r[1]], [0, 0]])
        else:
            mask_true = np.pad(mask_true, [[l[0], r[0]], [l[1], r[1]], [0, 0]])
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def crop(x_true, y_true = None, bbox_true = None, mask_true = None, bbox = None, min_area = 0., min_visibility = 0.):
    """
    bbox = [x1, y1, x2, y2]
    """
    if bbox is not None:
        result = albumentations(x_true, y_true, bbox_true, mask_true, min_area = min_area, min_visibility = min_visibility,
                                transform = [A.Crop(*bbox, always_apply = True)])
    else:
        result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
        result = result[0] if len(result) == 1 else tuple(result)
    return result

def random_apply(function, x_true, y_true = None, bbox_true = None, mask_true = None, p = 0.5, choice_size = 1, 
                 image_shape = None, max_pad_size = 100, pad_val = 0, mode = "right", background = "bg", **kwargs):
    """
    x_true = (N, H, W, C)
    y_true(without bbox_true) = (N, n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    """
    n_batch = len(x_true)
    indices = np.arange(n_batch)
    x_trues = []
    y_trues = []
    bbox_trues = []
    mask_trues = []
    for index in range(n_batch):
        if np.random.random() < p:
            if 1 < choice_size:
                #index = [index] + random.choices(indices, k = choice_size - 1)
                index = [index] + np.random.choice(indices, choice_size - 1).tolist() #for numpy seed
            out = function(x_true[index], 
                           y_true[index] if y_true is not None else None,
                           bbox_true[index] if bbox_true is not None else None, 
                           mask_true[index] if mask_true is not None else None, **kwargs)
            if not isinstance(out, tuple) and not isinstance(out, list):
                out = (out,)
        else:
            out = [arg[index] for arg in [x_true, y_true, bbox_true, mask_true] if arg is not None]
        args = {k:v for k, v in zip(["x_true", "y_true", "bbox_true", "mask_true"], out)}
        out = pad(**args, image_shape = image_shape, max_pad_size = max_pad_size, pad_val = pad_val, mode = mode, background = background)
        if not isinstance(out, tuple) and not isinstance(out, list):
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
        raise ValueError("all input arrays must have the same shape : please check 'image_shape' or 'max_pad_size'")
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result