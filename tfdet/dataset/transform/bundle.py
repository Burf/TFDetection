import functools

import cv2
import numpy as np

from .augment import *
from .common import *

try:
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
                          min_area = 0., min_visibility = 0., e = 1e-12):
        """
        x_true = (H, W, C)
        y_true(without bbox_true) = (1 or n_class)
        y_true(with bbox_true) = (P, 1 or n_class)
        bbox_true = (P, 4)
        mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
        mask_true(semantic mask_true) = (H, W, 1 or n_class)

        #albumentations > random_flip > random_crop(optional)
        #Pad is removed.
        #If crop_shape is shape or ratio, apply random_crop.
        """
        func_transform = [functools.partial(albumentations, transform = transform, min_area = min_area, min_visibility = min_visibility),
                          functools.partial(random_flip, p = p_flip, mode = flip_mode)]
        if crop_shape is not None:
            func_transform.append(functools.partial(random_crop, image_shape = crop_shape, min_area = min_area, min_visibility = min_visibility, e = e))
        return compose(x_true, y_true, bbox_true, mask_true, transform = func_transform)
except:
    pass    

def yolo_augmentation(x_true, y_true = None, bbox_true = None, mask_true = None,
                      image_shape = None, pad_val = 114,
                      perspective = 0., rotate = 0., translate = 0.2, scale = 0.9, shear = 0.,
                      h = 0.015, s = 0.7, v = 0.4,
                      max_paste_count = 20, scale_range = [0.0625, 0.75], clip_object = True, replace = True, random_count = False, label = None,
                      min_scale = 2, min_instance_area = 1, iou_threshold = 0.3, copy_min_scale = 2, copy_min_instance_area = 1, copy_iou_threshold = 0.3, p_copy_paste_flip = 0.5, method = cv2.INTER_LINEAR,
                      p_mosaic = 1., p_mix_up = 0.15, p_copy_paste = 0., p_flip = 0.5, p_mosaic9 = 0.8,
                      min_area = 0., min_visibility = 0., e = 1e-12):
    """
    https://github.com/WongKinYiu/yolov7/blob/main/utils/datasets.py
    
    x_true = (N, H, W, C)
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    #(mosaic + random_perspective > mix_up(with sample mosaic + random_perspective)) or (letter_box + random_perspective) > yolo_hsv > copy_paste(optional) > random_flip
    #Pad is removed.
    #First image is Background image.
    """
    if np.ndim(x_true[0]) < 3:
        x_true = np.expand_dims(x_true, axis = 0)
        y_true = np.expand_dims(y_true, axis = 0) if y_true is not None else None
        bbox_true = np.expand_dims(bbox_true, axis = 0) if bbox_true is not None else None
        mask_true = np.expand_dims(mask_true, axis = 0) if mask_true is not None else None
    
    indices = np.arange(len(x_true))
    
    keys = ["x_true", "y_true", "bbox_true", "mask_true"]
    values = [x_true, y_true, bbox_true, mask_true]
    kwargs = {k:v for k, v in zip(keys, values) if v is not None}
    target_kwargs = {k:np.array(v[0]) for k, v in kwargs.items()}
    
    image_shape = np.shape(target_kwargs["x_true"])[:2] if image_shape is None else image_shape
    if np.random.random() < p_mosaic:
        if np.random.random() < p_mosaic9:
            sample_indices = np.random.choice(indices, 3, replace = True)
            mosaic_func = functools.partial(mosaic, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e)
        else:
            sample_indices = np.random.choice(indices, 8, replace = True)
            mosaic_func = functools.partial(mosaic9, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e)
        sample_kwargs = {k:[target_kwargs[k]] + [v[i] for i in sample_indices] for k, v in kwargs.items()}
        sample_transform = [mosaic_func,
                            functools.partial(random_perspective, image_shape = image_shape, perspective = perspective, rotate = rotate, translate = translate, scale = scale, shear = shear, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e),
                            functools.partial(filter_annotation, min_scale = min_scale, min_instance_area = min_instance_area)]
        target_kwargs = compose(sample_kwargs, transform = sample_transform)  
        if np.random.random() < p_mix_up:
            if np.random.random() < p_mosaic9:
                sample_indices = np.random.choice(indices, 4, replace = True)
                mosaic_func = functools.partial(mosaic, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e)
            else:
                sample_indices = np.random.choice(indices, 9, replace = True)
                mosaic_func = functools.partial(mosaic9, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e)
            sample_kwargs = {k:[target_kwargs[k]] + [v[i] for i in sample_indices] for k, v in kwargs.items()}
            sample_transform = [mosaic_func,
                                functools.partial(random_perspective, image_shape = image_shape, perspective = perspective, rotate = rotate, translate = translate, scale = scale, shear = shear, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e),
                                functools.partial(filter_annotation, min_scale = min_scale, min_instance_area = min_instance_area)]
            sample_kwargs = compose(sample_kwargs, transform = sample_transform)
            sample_kwargs = {k:[target_kwargs[k], v] for k, v in sample_kwargs.items()}
            target_kwargs = compose(sample_kwargs, transform = mix_up)
    else:
        transform = [functools.partial(pad, image_shape = image_shape, max_pad_size = 0, pad_val = pad_val),
                     functools.partial(random_perspective, image_shape = image_shape, perspective = perspective, rotate = rotate, translate = translate, scale = scale, shear = shear, pad_val = pad_val, min_area = min_area, min_visibility = min_visibility, e = e),
                     functools.partial(filter_annotation, min_scale = min_scale, min_instance_area = min_instance_area)]
        target_kwargs = compose(target_kwargs, transform = transform)
        
    transform = functools.partial(yolo_hsv, h = h, s = s, v = v)
    target_kwargs = compose(target_kwargs, transform = transform)
    
    if np.random.random() < p_copy_paste:
        sample_kwargs = {k:[target_kwargs[k]] + list(v) for k, v in kwargs.items()}
        transform = functools.partial(copy_paste,
                                      max_paste_count = max_paste_count, scale_range = scale_range, clip_object = clip_object, replace = replace, random_count = random_count, label = label, min_scale = min_scale, min_instance_area = min_instance_area, iou_threshold = iou_threshold, copy_min_scale = copy_min_scale, copy_min_instance_area = copy_min_instance_area, copy_iou_threshold = copy_iou_threshold, p_flip = p_copy_paste_flip, method = method, 
                                      min_area = min_area, min_visibility = min_visibility, e = e)
        target_kwargs = compose(sample_kwargs, transform = transform)
    
    transform = functools.partial(random_flip, p = p_flip, mode = "horizontal")
    target_kwargs = compose(target_kwargs, transform = transform)
    
    result = list([target_kwargs[key] for key in keys if key in target_kwargs])
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def mmdet_augmentation(x_true, y_true = None, bbox_true = None, mask_true = None,
                       image_shape = [1333, 800], keep_ratio = True, crop_shape = None, p_flip = 0.5,
                       flip_mode = "horizontal", method = cv2.INTER_LINEAR, resize_mode = "jitter",
                       shape_divisor = 32, max_pad_size = 100, pad_val = 114, pad_mode = "both", background = "background",
                       min_area = 0., min_visibility = 0., e = 1e-12):
    """
    https://github.com/open-mmlab/mmdetection/blob/master/configs/_base_/datasets/coco_detection.py
    
    x_true = (H, W, C)
    y_true(without bbox_true) = (1 or n_class)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    
    #random_resize > random_crop(optional) > random_flip > pad(by shape_divisor)
    #If crop_shape is shape or ratio, apply random_crop.
    #Pad is removed.(by random crop)
    """
    func_transform = [functools.partial(resize, image_shape = image_shape, keep_ratio = keep_ratio, method = method, mode = resize_mode)]
    if crop_shape is not None:
        func_transform.append(functools.partial(random_crop, image_shape = crop_shape, min_area = min_area, min_visibility = min_visibility, e = e))
    func_transform.append(functools.partial(random_flip, p = p_flip, mode = flip_mode))
    func_transform.append(functools.partial(pad, shape_divisor = shape_divisor, max_pad_size = max_pad_size, pad_val = pad_val, mode = pad_mode, background = background))
    return compose(x_true, y_true, bbox_true, mask_true, transform = func_transform)