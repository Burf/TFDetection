import cv2
import numpy as np

from .augment import *
from .common import *

def yolo_augmentation(x_true, y_true = None, bbox_true = None, mask_true = None,
                      image_shape = None, pad_val = 114,
                      perspective = 0., rotate = 0., translate = 0.2, scale = 0.9, shear = 0.,
                      h = 0.015, s = 0.7, v = 0.4,
                      max_paste_count = 20, scale_range = [0.0625, 0.75], clip_object = True, replace = True, random_count = False, label = None,
                      min_scale = 2, min_instance_area = 1, iou_threshold = 0.3, copy_min_scale = 2, copy_min_instance_area = 1, copy_iou_threshold = 0.3, p_copy_paste_flip = 0.5, method = cv2.INTER_LINEAR,
                      p_mosaic = 1., p_mix_up = 0.15, p_copy_paste = 0., p_flip = 0.5,
                      min_area = 0., min_visibility = 0., e = 1e-12):
    """
    https://github.com/WongKinYiu/yolov7/blob/main/utils/datasets.py
    
    x_true = (N, H, W, C)
    y_true(without bbox_true) = (N, n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    #The pad will be removed.
    #First image is Background image.
    """
    if np.ndim(x_true[0]) < 3:
        x_true = np.expand_dims(x_true, axis = 0)
        y_true = np.expand_dims(y_true, axis = 0) if y_true is not None else None
        bbox_true = np.expand_dims(bbox_true, axis = 0) if bbox_true is not None else None
        mask_true = np.expand_dims(mask_true, axis = 0) if mask_true is not None else None
    
    indices = np.arange(len(x_true))
    
    image = np.array(x_true[0])
    y = np.array(y_true[0]) if y_true is not None else None
    bbox = np.array(bbox_true[0]) if bbox_true is not None else None
    mask = np.array(mask_true[0]) if mask_true is not None else None
    
    image_shape = np.shape(image)[:2] if image_shape is None else image_shape
    if np.random.random() < p_mosaic:
        if np.random.random() < 0.8:
            sample_indices = np.random.choice(indices, 3, replace = True)
            sample_x = [image] + [x_true[i] for i in sample_indices]
            sample_y = ([y] + [y_true[i] for i in sample_indices]) if y_true is not None else None
            sample_bbox = ([bbox] + [bbox_true[i] for i in sample_indices]) if bbox_true is not None else None
            sample_mask = ([mask] + [mask_true[i] for i in sample_indices]) if mask_true is not None else None
            out = mosaic(sample_x, sample_y, sample_bbox, sample_mask, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e)
        else:
            sample_indices = np.random.choice(indices, 8, replace = True)
            sample_x = [image] + [x_true[i] for i in sample_indices]
            sample_y = ([y] + [y_true[i] for i in sample_indices]) if y_true is not None else None
            sample_bbox = ([bbox] + [bbox_true[i] for i in sample_indices]) if bbox_true is not None else None
            sample_mask = ([mask] + [mask_true[i] for i in sample_indices]) if mask_true is not None else None
            out = mosaic9(sample_x, sample_y, sample_bbox, sample_mask, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility)
        out = [out] if not isinstance(out, tuple) else out
        out = random_perspective(*out, image_shape = image_shape, perspective = perspective, rotate = rotate, translate = translate, scale = scale, shear = shear, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e)
        out = [out] if not isinstance(out, tuple) else out
        
        if np.random.random() < p_mix_up:
            if np.random.random() < 0.8:
                sample_indices = np.random.choice(indices, 4, replace = True)
                sample_x = [x_true[i] for i in sample_indices]
                sample_y = [y_true[i] for i in sample_indices] if y_true is not None else None
                sample_bbox = [bbox_true[i] for i in sample_indices] if bbox_true is not None else None
                sample_mask = [mask_true[i] for i in sample_indices] if mask_true is not None else None
                out2 = mosaic(sample_x, sample_y, sample_bbox, sample_mask, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e)
            else:
                sample_indices = np.random.choice(indices, 9, replace = True)
                sample_x = [x_true[i] for i in sample_indices]
                sample_y = [y_true[i] for i in sample_indices] if y_true is not None else None
                sample_bbox = [bbox_true[i] for i in sample_indices] if bbox_true is not None else None
                sample_mask = [mask_true[i] for i in sample_indices] if mask_true is not None else None
                out2 = mosaic9(sample_x, sample_y, sample_bbox, sample_mask, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility)
            out2 = [out2] if not isinstance(out2, tuple) else out2
            out2 = random_perspective(*out2, image_shape = image_shape, perspective = perspective, rotate = rotate, translate = translate, scale = scale, shear = shear, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e)
            out2 = [out2] if not isinstance(out2, tuple) else out2
            out = mix_up(*[[o, o2] for o, o2 in zip(out, out2)])
            out = [out] if not isinstance(out, tuple) else out
    else:
        out = pad(image, y, bbox, mask, image_shape = image_shape, max_pad_size = 0, pad_val = pad_val)
        out = [out] if not isinstance(out, tuple) else out
        out = random_perspective(*out, image_shape = image_shape, perspective = perspective, rotate = rotate, translate = translate, scale = scale, shear = shear, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e)
        out = [out] if not isinstance(out, tuple) else out
        
    out = yolo_hsv(*out, h = h, s = s, v = v)
    out = [out] if not isinstance(out, tuple) else out
    
    if np.random.random() < p_copy_paste:
        out = [[o] + list(o2) for o, o2 in zip(out, [arg for arg in [x_true, y_true, bbox_true, mask_true] if arg is not None])]
        out = copy_paste(*out, 
                         max_paste_count = max_paste_count, scale_range = scale_range, clip_object = clip_object, replace = replace, random_count = random_count, label = label, min_scale = min_scale, min_instance_area = min_instance_area, iou_threshold = iou_threshold, copy_min_scale = copy_min_scale, copy_min_instance_area = copy_min_instance_area, copy_iou_threshold = copy_iou_threshold, p_flip = p_copy_paste_flip, method = method, 
                         min_area = min_area, min_visibility = min_visibility, e = e)
        out = [out] if not isinstance(out, tuple) else out
            
    out = random_flip(*out, p = p_flip)
    out = [out] if not isinstance(out, tuple) else out
    
    out = list(out)
    #x_true = np.array(out.pop(0), dtype = x_true[0].dtype)
    x_true = out.pop(0)
    if 0 < len(out):
        y_true = out.pop(0)
    if 0 < len(out):
        bbox_true = np.array(out.pop(0), dtype = bbox_true[0].dtype)
    if 0 < len(out):
        mask_true = out.pop(0)
    
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result