import random

import albumentations as A
import cv2
import numpy as np

from tfdet.core.bbox import random_bbox, overlap_bbox_numpy as overlap_bbox
from ..util.image import instance2bbox

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
                   min_area = 0., min_visibility = 0.):
    """
    x_true = (H, W, C)
    y_true(without bbox_true) = (1 or n_class)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    
    #The pad will be removed.
    """
    if 0 < len(transform):
        method = A.Compose(transform, bbox_params = {"format":"albumentations", "label_fields":["class_labels"], "min_area":min_area, "min_visibility":min_visibility})
        
        h, w = np.shape(x_true)[:2]
        class_labels = None
        if bbox_true is not None:
            bbox_true = np.array(bbox_true)
            indices = np.where(np.max(0 < bbox_true, axis = -1, keepdims = True) != 0)[0]
            bbox_true = bbox_true[indices]
            bbox_norm = not np.any(np.greater_equal(bbox_true, 2))
            if not bbox_norm:
                bbox_true = np.divide(bbox_true, [w, h, w, h])
            area = (bbox_true[..., 3] - bbox_true[..., 1]) * (bbox_true[..., 2] - bbox_true[..., 0])
            indices2 = np.where(0 < area)[0]
            bbox_true = bbox_true[indices2]
            if y_true is not None:
                y_true = np.array(y_true)[indices[indices2]]
            class_labels = np.arange(len(bbox_true))
        if mask_true is not None and 3 < np.ndim(mask_true):
            if bbox_true is not None:
                mask_true = np.array(mask_true)[indices[indices2]]
            if 0 < len(mask_true):
                mask_true = [np.array(m) for m in mask_true]
        
        args = {"image":x_true, "class_labels":class_labels if class_labels is not None else [0]}
        args["bboxes"] = bbox_true if bbox_true is not None else [[0, 0, 1, 1]]
        if mask_true is not None:
            if np.ndim(mask_true) < 4 or 0 < len(mask_true):
                args["masks" if 3 < np.ndim(mask_true) else "mask"] = mask_true
        
        aug_result = method(**args)
        indices = aug_result["class_labels"] if class_labels is not None else (np.arange(len(y_true)) if y_true is not None else [])
        x_true = aug_result["image"]
        ori_bbox_true = bbox_true
        if y_true is not None:
            y_true = np.array(y_true)[indices]
        if mask_true is not None:
            if np.ndim(mask_true) < 4 or 0 < len(mask_true):
                mask_true = np.array(aug_result["masks" if 3 < np.ndim(mask_true) else "mask"])
            if 3 < np.ndim(mask_true):
                if bbox_true is not None:
                    mask_true = mask_true[indices]
                #else:
                #    mask_true = mask_true[0 < np.max(mask_true, axis = (-3, -2, -1))]
        if bbox_true is not None:
            new_bbox_true = aug_result["bboxes"]
            if len(new_bbox_true) == 0:
                bbox_true = np.zeros((0, 4), dtype = bbox_true.dtype)
            else:
                bbox_true = np.clip(new_bbox_true, 0, 1)
                #area = (bbox_true[..., 3] - bbox_true[..., 1]) * (bbox_true[..., 2] - bbox_true[..., 0])
                #indices = np.where(0 < area)[0]
                #bbox_true = bbox_true[indices]
                bbox_true = bbox_true if bbox_norm else np.round(np.multiply(new_bbox_true, np.tile(np.shape(x_true)[:2][::-1], 2))).astype(int)
                #if y_true is not None:
                #    y_true = y_true[indices]
                #if mask_True is not None and 3 < np.ndim(mask_true):
                #    mask_true = mask_true[indices]
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result
    
def random_crop(x_true, y_true = None, bbox_true = None, mask_true = None, image_shape = None, min_area = 0., min_visibility = 0.):
    """
    x_true = (H, W, C)
    y_true(without bbox_true) = (1 or n_class)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    
    #The pad will be removed.
    """
    if image_shape is not None:
        if np.ndim(image_shape) == 0:
            image_shape = np.round(np.multiply(np.shape(x_true)[:2], image_shape)).astype(int)
        result = albumentations(x_true, y_true, bbox_true, mask_true, min_area = min_area, min_visibility = min_visibility,
                                transform = [A.RandomCrop(*image_shape[:2], always_apply = True)])
    else:
        result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
        result = result[0] if len(result) == 1 else tuple(result)
    return result

def random_flip(x_true, y_true = None, bbox_true = None, mask_true = None, p = 0.5, mode = "horizontal"):
    """
    x_true = (H, W, C)
    y_true(without bbox_true) = (1 or n_class)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    
    mode = ("horizontal", "vertical", "both")
    """
    if mode not in ("horizontal", "vertical", "both"):
        raise ValueError("unknown mode '{0}'".format(mode))
    if 0 < p:
        if mode == "horizontal":
            transform = [A.HorizontalFlip(p = p)]
        elif mode == "vertical":
            transform = [A.VerticalFlip(p = p)]
        else:
            transform = [A.HorizontalFlip(p = p), A.VerticalFlip(p = p)]
        result = albumentations(x_true, y_true, bbox_true, mask_true, transform = transform)
    else:
        result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
        result = result[0] if len(result) == 1 else tuple(result)
    return result

def yolo_hsv(x_true, y_true = None, bbox_true = None, mask_true = None, h = 0.015, s = 0.7, v = 0.4):
    """
    https://github.com/WongKinYiu/yolov7/blob/main/utils/datasets.py
    
    x_true = (H, W, C) #RGB, np.uint8
    y_true(without bbox_true) = (1 or n_class)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    """
    r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(np.array(x_true, dtype = np.uint8), cv2.COLOR_RGB2HSV))

    x = np.arange(256)
    lut_hue = ((x * r[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(np.uint8)
    x_true = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def random_perspective(x_true, y_true = None, bbox_true = None, mask_true = None, image_shape = None, perspective = 0., rotate = 0., translate = 0.2, scale = 0.9, shear = 0., pad_val = 114, min_area = 0., min_visibility = 0., e = 1e-12):
    """
    https://github.com/WongKinYiu/yolov7/blob/main/utils/datasets.py
    
    x_true = (H, W, C)
    y_true(without bbox_true) = (1 or n_class)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    
    #The pad will be removed.
    """
    if image_shape is not None and np.ndim(image_shape) == 0:
        image_shape = np.round(np.multiply(np.shape(x_true)[:2], image_shape)).astype(int)
    h, w = np.shape(x_true)[:2]
    new_h, new_w = (h, w) if image_shape is None else image_shape[:2]
    
    # Center
    C = np.eye(3)
    C[0, 2] = -w / 2  # x translation (pixels)
    C[1, 2] = -h / 2  # y translation (pixels)
    
    # Perspective
    P = np.eye(3)
    P[2, 0] = np.random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = np.random.uniform(-perspective, perspective)  # y perspective (about x)
    
    # Rotation and Scale
    R = np.eye(3)
    a = np.random.uniform(-rotate, rotate)
    # a += np.random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = np.random.uniform(1 - scale, 1.1 + scale)
    # s = 2 ** np.random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle = a, center = (0, 0), scale = s)
    
    # Shear
    S = np.eye(3)
    S[0, 1] = np.math.tan(np.random.uniform(-shear, shear) * np.math.pi / 180)  # x shear (deg)
    S[1, 0] = np.math.tan(np.random.uniform(-shear, shear) * np.math.pi / 180)  # y shear (deg)
    
    # Translation
    T = np.eye(3)
    T[0, 2] = np.random.uniform(0.5 - translate, 0.5 + translate) * new_w  # x translation (pixels)
    T[1, 2] = np.random.uniform(0.5 - translate, 0.5 + translate) * new_h  # y translation (pixels)
    
    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (h != new_h) or (w != new_w) or (M != np.eye(3)).any():  # image changed
        if perspective != 0:
            x_true = cv2.warpPerspective(x_true, M, dsize = (new_w, new_h), borderValue = (pad_val,) * np.shape(x_true)[-1])
        else:  # affine
            x_true = cv2.warpAffine(x_true, M[:2], dsize = (new_w, new_h), borderValue = (pad_val,) * np.shape(x_true)[-1])
        
        if bbox_true is not None:
            bbox_true = np.array(bbox_true)
            bbox_norm = not np.any(np.greater_equal(bbox_true, 2))
            if bbox_norm:
                bbox_true = np.multiply(bbox_true, [w, h, w, h])
            
            xy = np.ones((len(bbox_true) * 4, 3))
            xy[:, :2] = bbox_true[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(len(bbox_true) * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective != 0 else xy[:, :2]).reshape(len(bbox_true), 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new_bbox = np.concatenate([np.min(x, axis = 1), np.min(y, axis = 1), np.max(x, axis = 1), np.max(y, axis = 1)]).reshape(4, len(bbox_true)).T

            # clip
            new_bbox[:, [0, 2]] = np.clip(new_bbox[:, [0, 2]], 0, new_w)
            new_bbox[:, [1, 3]] = np.clip(new_bbox[:, [1, 3]], 0, new_h)
            
            area = (bbox_true[..., 2] - bbox_true[..., 0]) * (bbox_true[..., 3] - bbox_true[..., 1])
            new_area = (new_bbox[..., 2] - new_bbox[..., 0]) * (new_bbox[..., 3] - new_bbox[..., 1])
            flag = np.logical_and(min_area <= (new_area / (new_w * new_h)), min_visibility < (new_area / (area + e)))
            
            bbox_true = np.divide(new_bbox, [new_w, new_h, new_w, new_h]) if bbox_norm else np.round(new_bbox).astype(int)
            bbox_true = bbox_true[flag]
            if y_true is not None:
                y_true = np.array(y_true)[flag]
        
        if mask_true is not None:
            if 3 < np.ndim(mask_true) and len(mask_true) == 0:
                mask_true = np.array(mask_true)
                mask_true = np.zeros((0, new_h, new_w, np.shape(mask_true)[-1]), dtype = mask_true.dtype)
            else:
                masks = []
                for mask in mask_true if 3 < np.ndim(mask_true) else [mask_true]:
                    if perspective != 0:
                        new_mask = cv2.warpPerspective(mask, M, dsize = (new_w, new_h))
                    else:  # affine
                        new_mask = cv2.warpAffine(mask, M[:2], dsize = (new_w, new_h))
                    new_mask = np.expand_dims(new_mask, axis = -1) if np.ndim(mask) != np.ndim(new_mask) else new_mask
                    masks.append(new_mask)

                if 0 < len(masks):
                    mask_true = np.stack(masks, axis = 0) if 3 < np.ndim(mask_true) else masks[0]
                    if 3 < np.ndim(mask_true):
                        #mask_true = mask_true[0 < np.max(mask_true, axis = (-3, -2, -1))] if bbox_true is None else mask_true[flag]
                        mask_true = mask_true[flag] if bbox_true is not None else mask_true
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result
        
def mosaic(x_true, y_true = None, bbox_true = None, mask_true = None, image_shape = None, alpha = 0.25, pad_val = 114, min_area = 0., min_visibility = 0., e = 1e-12):
    """
    https://github.com/WongKinYiu/yolov7/blob/main/utils/datasets.py
    
    x_true = (4, H, W, C)
    y_true(without bbox_true) = (4, n_class)
    y_true(with bbox_true) = (4, P, 1 or n_class)
    bbox_true = (4, P, 4)
    mask_true(with bbox_true & instance mask_true) = (4, P, H, W, 1)
    mask_true(semantic mask_true) = (4, H, W, 1 or n_class)
    
    #The pad will be removed.
    #If image_shape is None, the result is (N, 2 * H, 2 * W, C).
    """
    if np.ndim(x_true[0]) < 3:
        x_true = np.expand_dims(x_true, axis = 0)
        y_true = np.expand_dims(y_true, axis = 0) if y_true is not None else None
        bbox_true = np.expand_dims(bbox_true, axis = 0) if bbox_true is not None else None
        mask_true = np.expand_dims(mask_true, axis = 0) if mask_true is not None else None
    if image_shape is not None and np.ndim(image_shape) == 0:
        image_shape = np.round(np.multiply(np.shape(x_true[0])[:2], image_shape)).astype(int)
    
    h, w, c = np.shape(x_true[0]) if 2 < np.ndim(x_true[0]) else [*np.shape(x_true[0]), 1]
    if image_shape is not None:
        h, w, c = image_shape if 2 < len(image_shape) else [*image_shape, c]
    else:
        h, w = h * 2, w * 2
    image = np.full([h, w, c], pad_val, dtype = x_true[0].dtype)
    center = np.round(np.multiply([np.random.uniform(alpha, 1 - alpha), np.random.uniform(alpha, 1 - alpha)], [w, h])).astype(int)
    pads = []
    masks = None
    for i in range(min(len(x_true), 4)):
        img = x_true[i] if 2 < np.ndim(x_true[i]) else np.expand_dims(x_true[i], axis = -1)
        img_h, img_w = np.shape(img)[:2]
        if i == 0: #top left
            x1a, y1a, x2a, y2a = max(center[0] - img_w, 0), max(center[1] - img_h, 0), center[0], center[1] #xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = img_w - (x2a - x1a), img_h - (y2a - y1a), img_w, img_h  #xmin, ymin, xmax, ymax (small image)
        elif i == 1: #top right
            x1a, y1a, x2a, y2a = center[0], max(center[1] - img_h, 0), min(center[0] + img_w, w), center[1]
            x1b, y1b, x2b, y2b = 0, img_h - (y2a - y1a), min(img_w, x2a - x1a), img_h
        elif i == 2: #bottom left
            x1a, y1a, x2a, y2a = max(center[0] - img_w, 0), center[1], center[0], min(h, center[1] + img_h)
            x1b, y1b, x2b, y2b = img_w - (x2a - x1a), 0, img_w, min(y2a - y1a, img_h)
        elif i == 3: #bottom right
            x1a, y1a, x2a, y2a = center[0], center[1], min(center[0] + img_w, w), min(h, center[1] + img_h)
            x1b, y1b, x2b, y2b = 0, 0, min(img_w, x2a - x1a), min(y2a - y1a, img_h)
        image[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        pads.append([x1a - x1b, y1a - y1b])
        if mask_true is not None:
            if np.ndim(mask_true[i]) < 4: #semantic_mask
                if i == 0:
                    masks = np.zeros([h, w, np.shape(mask_true[i])[-1]], dtype = mask_true[i].dtype)
                masks[..., y1a:y2a, x1a:x2a, :] = mask_true[i][..., y1b:y2b, x1b:x2b, :]
            elif 3 < np.ndim(mask_true[i]): #instance_mask
                mask = np.array(mask_true[i])
                if i == 0:
                    masks = []
                new_mask = np.zeros([len(mask), h, w, 1], dtype = mask.dtype)
                new_mask[..., y1a:y2a, x1a:x2a, :] = mask[..., y1b:y2b, x1b:x2b, :]
                #if bbox_true is None:
                #    new_mask = new_mask[0 < np.max(new_mask, axis = (-3, -2, -1))]
                masks.append(new_mask)
    
    if bbox_true is None and y_true is not None:
        h, w = np.shape(image)[:2]
        area = [center[0] * center[1], (w - center[0]) * center[1], center[0] * (h - center[1]), (w - center[0]) * (h - center[1])][:min(4, len(x_true))]
        ratio = np.divide(area, sum(area))
        y_true = np.sum([np.multiply(l, r) for l, r in zip(y_true, ratio)], axis = 0)
    elif bbox_true is not None:
        norm = True
        for bbox in bbox_true[:min(len(x_true), 4)]:
            if np.any(np.greater_equal(bbox, 2)):
                norm = False
                break
        if norm:
            pads = np.divide(pads, [w, h])
            h = w = 1
        bboxes = []
        labels = []
        for i, bbox in enumerate(bbox_true[:min(len(x_true), 4)]):
            bbox = np.divide(np.multiply(bbox, np.tile(np.shape(x_true[i])[:2][::-1], 2)), np.tile(np.shape(image)[:2][::-1], 2)) if norm else np.array(bbox)
            new_bbox = np.add(bbox, np.tile(pads[i], 2))
            new_bbox = np.clip(new_bbox, 0, [w, h, w, h])
            
            area = (bbox[..., 2] - bbox[..., 0]) * (bbox[..., 3] - bbox[..., 1])
            new_area = (new_bbox[..., 2] - new_bbox[..., 0]) * (new_bbox[..., 3] - new_bbox[..., 1])
            flag = np.logical_and(min_area <= (new_area / (w * h)), min_visibility < (new_area / (area + e)))
            
            new_bbox = new_bbox[flag]
            bboxes.append(new_bbox)
            if y_true is not None:
                labels.append(np.array(y_true[i])[flag])
            if isinstance(masks, list):
                masks[i] = masks[i][flag]
        
        bbox_true = np.concatenate(bboxes, axis = 0) if 0 < len(bboxes) else np.zeros((0, 4), dtype = np.array(bbox_true).dtype)
        if y_true is not None:
            y_true = np.concatenate(labels, axis = 0) if 0 < len(labels) else np.zeros((0, np.shape(y_true)[-1]), dtype = np.array(y_true).dtype)
    if isinstance(masks, list):
        masks = np.concatenate(masks, axis = 0) if 0 < len(masks) else np.zeros((0, h, w, 1), dtype = np.array(mask_true).dtype)
    x_true, mask_true = image, masks
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def mosaic9(x_true, y_true = None, bbox_true = None, mask_true = None, image_shape = None, pad_val = 114, min_area = 0., min_visibility = 0.):
    """
    https://github.com/WongKinYiu/yolov7/blob/main/utils/datasets.py
    
    x_true = (9, H, W, C)
    y_true(without bbox_true) = (9, n_class)
    y_true(with bbox_true) = (9, P, 1 or n_class)
    bbox_true = (9, P, 4)
    mask_true(with bbox_true & instance mask_true) = (9, P, H, W, 1)
    mask_true(semantic mask_true) = (9, H, W, 1 or n_class)
    
    #The pad will be removed.
    #If image_shape is None, the result is (N, 2 * H, 2 * W, C).
    """
    if np.ndim(x_true[0]) < 3:
        x_true = np.expand_dims(x_true, axis = 0)
        y_true = np.expand_dims(y_true, axis = 0) if y_true is not None else None
        bbox_true = np.expand_dims(bbox_true, axis = 0) if bbox_true is not None else None
        mask_true = np.expand_dims(mask_true, axis = 0) if mask_true is not None else None
    if image_shape is not None and np.ndim(image_shape) == 0:
        image_shape = np.round(np.multiply(np.shape(x_true[0])[:2], image_shape)).astype(int)
        
    h, w, c = np.shape(x_true[0]) if 2 < np.ndim(x_true[0]) else [*np.shape(x_true[0]), 1]
    image_shape = [h * 2, w * 2] if image_shape is None else image_shape
    if 2 < len(image_shape):
        c = image_shape[2]
    image = np.full([h * 3, w * 3, c], pad_val, dtype = x_true[0].dtype)
    if np.any(np.greater(image_shape[:2], np.shape(image)[:2])):
        raise ValueError("'image_shape' should be less than or equal to ({0}, {1}). : please check 'image_shape'".format(*np.shape(image)[:2]))
    pads = []
    masks = None
    for i in range(min(len(x_true), 9)):
        img = x_true[i] if 2 < np.ndim(x_true[i]) else np.expand_dims(x_true[i], axis = -1)
        img_h, img_w = np.shape(img)[:2]
        
        if i == 0:  # center
            c = w, h, w + img_w, h + img_h  # xmin, ymin, xmax, ymax (base) coordinates
            img_h0, img_w0 = img_h, img_w
        elif i == 1:  # top
            c = w, h - img_h, w + img_w, h
        elif i == 2:  # top right
            c = w + img_wp, h - img_h, w + img_wp + img_w, h
        elif i == 3:  # right
            c = w + img_w0, h, w + img_w0 + img_w, h + img_h
        elif i == 4:  # bottom right
            c = w + img_w0, h + img_hp, w + img_w0 + img_w, h + img_hp + img_h
        elif i == 5:  # bottom
            c = w + img_w0 - img_w, h + img_h0, w + img_w0, h + img_h0 + img_h
        elif i == 6:  # bottom left
            c = w + img_w0 - img_wp - img_w, h + img_h0, w + img_w0 - img_wp, h + img_h0 + img_h
        elif i == 7:  # left
            c = w - img_w, h + img_h0 - img_h, w, h + img_h0
        elif i == 8:  # top left
            c = w - img_w, h + img_h0 - img_hp - img_h, w, h + img_h0 - img_hp
        
        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        image[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]
        img_hp, img_wp = img_h, img_w
        pads.append([padx, pady])
        
        if mask_true is not None:
            if np.ndim(mask_true[i]) < 4: #semantic_mask
                if i == 0:
                    masks = np.zeros([h * 3, w * 3, np.shape(mask_true[i])[-1]], dtype = mask_true[i].dtype)
                masks[..., y1:y2, x1:x2, :] = mask_true[i][..., y1 - pady:, x1 - padx:, :]
            elif 3 < np.ndim(mask_true[i]): #instance_mask
                mask = np.array(mask_true[i])
                if i == 0:
                    masks = []
                new_mask = np.zeros([len(mask), h * 3, w * 3, 1], dtype = mask.dtype)
                new_mask[..., y1:y2, x1:x2, :] = mask[..., y1 - pady:, x1 - padx:, :]
                #if bbox_true is None:
                #    new_mask = new_mask[0 < np.max(new_mask, axis = (-3, -2, -1))]
                masks.append(new_mask)
    
    if bbox_true is None and y_true is not None:
        h, w = np.divide(np.shape(image)[:2], 3).astype(int)
        clsf_y_true = np.arange(9).reshape((9, 1))[:len(y_true)]
        clsf_bbox_true = np.array([[h, w, 2 * h, 2 * w],
                                   [w, 0, 2 * w, h],
                                   [2 * w, 0, 3 * w, h],
                                   [2 * w, h, 3 * w, 2 * h],
                                   [2 * w, 2 * h, 3 * w, 3 * h],
                                   [w, 2 * h, 2 * w, 3 * h],
                                   [0, 2 * h, w, 3 * h],
                                   [0, h, w, 2 * h],
                                   [0, 0, w, h]])[:len(y_true)]
    elif bbox_true is not None:
        norm = True
        for bbox in bbox_true[:min(len(x_true), 9)]:
            if np.any(np.greater_equal(bbox, 2)):
                norm = False
                break
        if norm:
            pads = np.divide(pads, [w, h])
            h = w = 1
        bboxes = []
        labels = []
        for i, bbox in enumerate(bbox_true[:min(len(x_true), 9)]):
            indices = np.where(np.max(0 < bbox, axis = -1, keepdims = True) != 0)[0]
            bbox = bbox[indices]
            bbox = np.divide(np.multiply(bbox, np.tile(np.shape(x_true[i])[:2][::-1], 2)), np.tile(np.shape(image)[:2][::-1], 2)) if norm else np.array(bbox)
            bboxes.append(np.add(bbox, np.tile(pads[i], 2)))
            if y_true is not None:
                labels.append(np.array(y_true[i][indices]))
            if isinstance(masks, list) and i < len(masks):
                masks[i] = masks[i][indices]
            
        bbox_true = np.concatenate(bboxes, axis = 0) if 0 < len(bboxes) else np.zeros((0, 4), dtype = np.array(bbox_true).dtype)
        if y_true is not None:
            y_true = np.concatenate(labels, axis = 0) if 0 < len(labels) else np.zeros((0, np.shape(y_true)[-1]), dtype = np.array(y_true).dtype)
    if isinstance(masks, list):
        masks = np.concatenate(masks, axis = 0) if 0 < len(masks) else np.zeros((0, h, w, 1), dtype = np.array(mask_true).dtype)
    x_true, mask_true = image, masks
    
    if bbox_true is None and y_true is not None:
        x_true, clsf_y_true, clsf_bbox_true = random_crop(x_true, clsf_y_true, clsf_bbox_true, mask_true, image_shape = image_shape[:2], min_visibility = 0., min_area = 0.)
        area = (clsf_bbox_true[..., 3] - clsf_bbox_true[..., 1]) * clsf_bbox_true[..., 2] - clsf_bbox_true[..., 0]
        ratio = np.divide(area, sum(area))
        y_true = np.sum([np.multiply(y_true[l], ratio[i]) for i, l in enumerate(clsf_y_true[:, 0])], axis = 0)
        return (x_true, y_true)
    else:
        return random_crop(x_true, y_true, bbox_true, mask_true, image_shape = image_shape[:2], min_visibility = min_visibility, min_area = min_area)
    
def cut_mix(x_true, y_true = None, bbox_true = None, mask_true = None, alpha = 1., min_area = 0., min_visibility = 0., e = 1e-12):
    """
    x_true = (2, H, W, C)
    y_true(without bbox_true) = (2, n_class)
    y_true(with bbox_true) = (2, P, 1 or n_class)
    bbox_true = (2, P, 4)
    mask_true(with bbox_true & instance mask_true) = (2, P, H, W, 1)
    mask_true(semantic mask_true) = (2, H, W, 1 or n_class)
    
    #The pad will be removed.
    """
    if np.ndim(x_true[0]) < 3:
        x_true = np.expand_dims(x_true, axis = 0)
        y_true = np.expand_dims(y_true, axis = 0) if y_true is not None else None
        bbox_true = np.expand_dims(bbox_true, axis = 0) if bbox_true is not None else None
        mask_true = np.expand_dims(mask_true, axis = 0) if mask_true is not None else None
    
    if len(x_true) == 1:
        x_true = x_true[0]
        if mask_true is not None:
            mask_true = np.array(mask_true[0])
            #if 3 < np.ndim(mask_true) and bbox_true is None:
            #    mask_true = mask_true[0 < np.max(mask_true, axis = (-3, -2, -1))]
        if bbox_true is None and y_true is not None:
            y_true = y_true[0]
        elif bbox_true is not None:
            h, w = np.shape(x_true)[:2]
            bbox_true = np.array(bbox_true[0])
            indices = np.where(np.max(0 < bbox_true, axis = -1, keepdims = True) != 0)[0]
            bbox_true = bbox_true[indices]
            norm_bbox = np.divide(bbox_true, [w, h, w, h]) if not np.any(np.greater_equal(bbox_true, 2)) else bbox_true
            area = (norm_bbox[..., 3] - norm_bbox[..., 1]) * (norm_bbox[..., 2] - norm_bbox[..., 0])
            indices2 = np.where(min_area <= area)[0]
            bbox_true = bbox_true[indices2]
            if y_true is not None:
                y_true = np.array(y_true[0])[indices[indices2]]
            if mask_true is not None and 3 < np.ndim(mask_true):
                mask_true = mask_true[indices[indices2]]
    else:
        image = np.array(x_true[0])
        image_shape = image.shape[:2]
        crop_bbox = random_bbox(alpha, image_shape)
        image[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]] = x_true[1][crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]
        masks = None
        if mask_true is not None:
            if np.ndim(mask_true[0]) < 4:
                masks = np.array(mask_true[0])
                masks[..., crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :] = mask_true[1][..., crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :]
            elif 3 < np.ndim(mask_true[0]):
                masks = []
                for i in range(2):
                    if i == 0:
                        mask = np.array(mask_true[i])
                        mask[..., crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :] = 0
                    else:
                        m = np.array(mask_true[i])
                        mask = np.zeros([len(m), *image_shape, 1], dtype = m.dtype)
                        mask[..., crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :] = m[..., crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :]
                    if bbox_true is None:
                        mask = mask[0 < np.max(mask, axis = (-3, -2, -1))]
                    masks.append(mask)
        if bbox_true is None and y_true is not None:
            y_true = np.multiply(y_true[0], r) + np.multiply(y_true[1], 1 - r)
        elif bbox_true is not None:
            src_bbox, dst_bbox = np.array(bbox_true[0]), np.array(bbox_true[1])
            scale_crop_bbox = crop_bbox
            if not np.any(np.greater_equal(np.concatenate([src_bbox, dst_bbox]), 2)):
                scale_crop_bbox = crop_bbox / np.tile(image_shape[:2][::-1], 2)
                image_shape = (1, 1)

            src_area = (src_bbox[..., 2] - src_bbox[..., 0]) * (src_bbox[..., 3] - src_bbox[..., 1])
            src_inter = np.maximum(np.minimum(src_bbox[..., 2], scale_crop_bbox[2]) - np.maximum(src_bbox[..., 0], scale_crop_bbox[0]), 0) * np.maximum(np.minimum(src_bbox[..., 3], scale_crop_bbox[3]) - np.maximum(src_bbox[..., 1], scale_crop_bbox[1]), 0)
            new_area = src_area - src_inter
            src_flag = np.logical_and(min_area <= (new_area / (image_shape[0] * image_shape[1])), min_visibility < (new_area / (src_area + e)))

            dst_area = (dst_bbox[..., 2] - dst_bbox[..., 0]) * (dst_bbox[..., 3] - dst_bbox[..., 1])
            dst_bbox[..., [0, 2]] = np.clip(dst_bbox[..., [0, 2]], scale_crop_bbox[0], scale_crop_bbox[2])
            dst_bbox[..., [1, 3]] = np.clip(dst_bbox[..., [1, 3]], scale_crop_bbox[1], scale_crop_bbox[3])
            new_area = (dst_bbox[..., 2] - dst_bbox[..., 0]) * (dst_bbox[..., 3] - dst_bbox[..., 1])
            dst_flag = np.logical_and(min_area <= (new_area / (image_shape[0] * image_shape[1])), min_visibility < (new_area / (dst_area + e)))

            bbox_true = np.concatenate([src_bbox[src_flag], dst_bbox[dst_flag]], axis = 0)
            if y_true is not None:
                y_true = np.concatenate([np.array(y_true[0])[src_flag], np.array(y_true[1])[dst_flag]], axis = 0)
            if isinstance(masks, list):
                masks[0], masks[1] = masks[0][src_flag], masks[1][dst_flag]
        if isinstance(masks, list):
            masks = np.concatenate(masks, axis = 0)
        x_true, mask_true = image, masks
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result
    
def cut_out(x_true, y_true = None, bbox_true = None, mask_true = None, alpha = 1., pad_val = 114, min_area = 0., min_visibility = 0., e = 1e-12):
    """
    x_true = (H, W, C)
    y_true(without bbox_true) = (n_class)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    
    #The pad will be removed.
    """
    x_true = np.array(x_true)
    image_shape = x_true.shape[:2]
    crop_bbox = random_bbox(alpha, image_shape)
    x_true[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]] = pad_val
    if mask_true is not None and bbox_true is None:
        mask_true = np.array(mask_true)
        mask_true[..., crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :] = pad_val
        #if 3 < np.ndim(mask_true):
        #    mask_true = mask_true[0 < np.max(mask_true, axis = (-3, -2, -1))]
    if bbox_true is None and y_true is not None:
        y_true = np.array(y_true)
    elif bbox_true is not None:
        bbox_true = np.array(bbox_true)
        scale_crop_bbox = crop_bbox
        if not np.any(np.greater_equal(bbox_true, 2)):
            scale_crop_bbox = crop_bbox / np.tile(image_shape[:2][::-1], 2)
            image_shape = (1, 1)
        src_area = (bbox_true[..., 2] - bbox_true[..., 0]) * (bbox_true[..., 3] - bbox_true[..., 1])
        src_inter = np.maximum(np.minimum(bbox_true[..., 2], scale_crop_bbox[2]) - np.maximum(bbox_true[..., 0], scale_crop_bbox[0]), 0) * np.maximum(np.minimum(bbox_true[..., 3], scale_crop_bbox[3]) - np.maximum(bbox_true[..., 1], scale_crop_bbox[1]), 0)
        new_area = src_area - src_inter
        flag = np.logical_and(min_area <= (new_area / (image_shape[0] * image_shape[1])), min_visibility < (new_area / (src_area + e)))
        bbox_true = bbox_true[flag]
        if y_true is not None:
            y_true = np.array(y_true)[flag]
        if mask_true is not None:
            mask_true = np.array(mask_true)
            if 3 < np.ndim(mask_true):
                mask_true = mask_true[flag]
            mask_true[..., crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :] = pad_val
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def mix_up(x_true, y_true = None, bbox_true = None, mask_true = None, alpha = 8.):
    """
    x_true = (2, H, W, C)
    y_true(without bbox_true) = (2, n_class)
    y_true(with bbox_true) = (2, P, 1 or n_class)
    bbox_true = (2, P, 4)
    mask_true(with bbox_true & instance mask_true) = (2, P, H, W, 1)
    mask_true(semantic mask_true) = (2, H, W, 1 or n_class)
    
    #The pad will be removed.
    """
    if np.ndim(x_true[0]) < 3:
        x_true = np.expand_dims(x_true, axis = 0)
        y_true = np.expand_dims(y_true, axis = 0) if y_true is not None else None
        bbox_true = np.expand_dims(bbox_true, axis = 0) if bbox_true is not None else None
        mask_true = np.expand_dims(mask_true, axis = 0) if mask_true is not None else None
    
    if 1 < len(x_true):
        r = np.random.beta(alpha, alpha)
        x_true = [np.multiply(x_true[0], r) + np.multiply(x_true[1], 1 - r)]
        if bbox_true is None and y_true is not None:
            y_true = [np.multiply(y_true[0], r) + np.multiply(y_true[1], 1 - r)]
        elif bbox_true is not None:
            y_true = [np.concatenate(y_true[:2], axis = 0)]
            bbox_true = [np.concatenate(bbox_true[:2], axis = 0)]
        if mask_true is not None:
            if np.ndim(mask_true[0]) < 4:
                mask_true = [np.multiply(mask_true[0], r) + np.multiply(mask_true[1], 1 - r)]
            elif 3 < np.ndim(mask_true[0]):
                mask_true = [np.concatenate(mask_true[:2], axis = 0)]
    if len(x_true) == 1:
        x_true = x_true[0]
        if mask_true is not None:
            mask_true = np.array(mask_true[0])
            #if 3 < np.ndim(mask_true) and bbox_true is None:
            #    mask_true = mask_true[0 < np.max(mask_true, axis = (-3, -2, -1))]
        if bbox_true is None and y_true is not None:
            y_true = y_true[0]
        elif bbox_true is not None:
            #h, w = np.shape(x_true)[:2]
            bbox_true = np.array(bbox_true[0])
            indices = np.where(np.max(0 < bbox_true, axis = -1, keepdims = True) != 0)[0]
            bbox_true = bbox_true[indices]
            #norm_bbox = np.divide(bbox_true, [w, h, w, h]) if not np.any(np.greater_equal(bbox_true, 2)) else bbox_true
            #area = (norm_bbox[..., 3] - norm_bbox[..., 1]) * (norm_bbox[..., 2] - norm_bbox[..., 0])
            #indices2 = np.where(0 < area)[0]
            #bbox_true = bbox_true[indices2]
            if y_true is not None:
                #y_true = np.array(y_true[0])[indices[indices2]]
                y_true = np.array(y_true[0])[indices]
            if mask_true is not None and 3 < np.ndim(mask_true):
                #mask_true = mask_true[indices[indices2]]
                mask_true = mask_true[indices]
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def insta_boost(x_true, y_true = None, bbox_true = None, mask_true = None, 
                action_candidate = ("normal", "horizontal", "skip"), action_prob = (1, 0, 0), 
                scale = (0.8, 1.2), dx = 15, dy = 15, theta = (-1, 1),
                color_prob = 0.5, hflag = False):
    """
    https://arxiv.org/abs/1908.07801
    https://github.com/GothicAi/InstaBoost-pypi
    
    x_true = (H, W, C) #np.uint8
    y_true(without bbox_true) = (1 or n_class)
    y_true(with bbox_true) = (P, 1)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    
    #The pad will be removed.
    """
    if y_true is not None and bbox_true is not None and (mask_true is not None and 3 < np.ndim(mask_true)):
        try:
            import instaboostfast as instaboost
        except Exception as e:
            print("If you want to use 'insta_boost', please install 'instaboostfast'")
            raise e
            
        if 1 < np.shape(y_true)[-1]:
            y_true = np.expand_dims(np.argmax(y_true, axis = -1), axis = -1)
        indices = np.where(np.max(np.greater(bbox_true, 0), axis = -1, keepdims = True) != 0)[0]
        y_true = np.array(y_true)[indices]
        bbox_true = np.array(bbox_true)[indices]
        mask_true = np.array(mask_true)[indices]
        
        bbox_dtype = bbox_true.dtype
        mask_shape = np.shape(mask_true)[1:]
        mask_dtype = mask_true.dtype
        
        annos = []
        for y, bbox, mask in zip(y_true, bbox_true, mask_true):
            x1, y1, x2, y2 = bbox
            bbox = [x1, y1, x2 - x1, y2 - y1]
            anns.append({"category_id":y, "bbox":bbox, "segmentation":mask})

        config = instaboost.InstaBoostConfig(action_candidate, action_prob,
                                             scale, dx, dy, theta,
                                             color_prob, hflag)

        annos, x_true = instaboost.get_new_data(annos, x_true, config, background = None)
        
        y_true = []
        bbox_true = []
        mask_true = []
        for anno in annos:
            x1, y1, w, h = anno["bbox"]
            if w <= 0 or h <= 0:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            y_true.append(anno["category_id"])
            bbox_true.append(bbox)
            mask_true.append(anno["segmentation"])
        
        y_true = np.array(y_true) if 0 < len(y_true) else np.zeros((0, 1), dtype = np.int32)
        bbox_true = np.array(bbox_true) if 0 < len(bbox_true) else np.zeros((0, 4), dtype = bbox_dtype)
        mask_true = np.array(mask_true) if 0 < len(bbox_true) else np.zeros((0, *mask_shape), dtype = mask_dtype)
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def copy_paste(x_true, y_true = None, bbox_true = None, mask_true = None, max_paste_count = 100, scale_range = [0.0625, 0.75], clip_object = True, replace = True, random_count = True, label = None,
               min_scale = 2, min_instance_area = 1, iou_threshold = 0.3,
               copy_min_scale = 2, copy_min_instance_area = 1, copy_iou_threshold = 0.3,
               p_flip = 0.5, pad_val = 114, method = cv2.INTER_LINEAR,
               min_area = 0., min_visibility = 0., e = 1e-12):
    """
    https://arxiv.org/abs/2012.07177
    
    x_true = (N, H, W, C)
    y_true(without bbox_true) = (N, 1 or n_class)
    y_true(with bbox_true) = (N, P, 1 or n_class)
    bbox_true = (N, P, 4)
    mask_true(with bbox_true & instance mask_true) = (N, P, H, W, 1)
    mask_true(semantic mask_true) = (N, H, W, 1 or n_class)
    
    #The pad will be removed.
    #First image is Background image.
    #Paste object condition : min_scale[0] or min_scale <= paste_object_height and min_scale[1] or min_scale <= paste_object_width
    #Paste mask condition : min_instance_area <= paste_instance_mask_area
    scale = np.random.beta(1, 1.4) * np.abs(scale_range[1] - scale_range[0]) + np.min(scale_range)
    clip_object = Don't crop object
    replace = np.random.choice's replace for paste
    random_count = change max_paste_count from 0 to max_paste_count
    label = copy target label
    """
    if np.ndim(x_true[0]) < 3:
        x_true = np.expand_dims(x_true, axis = 0)
        y_true = np.expand_dims(y_true, axis = 0) if y_true is not None else None
        bbox_true = np.expand_dims(bbox_true, axis = 0) if bbox_true is not None else None
        mask_true = np.expand_dims(mask_true, axis = 0) if mask_true is not None else None
    
    if bbox_true is not None or (mask_true is not None and 3 < np.ndim(mask_true[0])):
        max_paste_count = np.random.randint(0, max_paste_count + 1) if random_count else max_paste_count
        min_scale = [min_scale, min_scale] if np.ndim(min_scale) == 0 else min_scale
        copy_min_scale = [copy_min_scale, copy_min_scale] if np.ndim(copy_min_scale) == 0 else copy_min_scale
        if bbox_true is not None:
            bbox_norm = True
            for bbox in bbox_true:
                if 0 < len(bbox) and np.any(np.greater_equal(bbox, 2)):
                    bbox_norm = False
                    break
            
        #copy
        shuffle_indices = np.arange(len(x_true))
        if 1 < len(x_true):
            shuffle_indices[1:]
        np.random.shuffle(shuffle_indices)
        sample_x, sample_y, sample_mask = [], [], []
        sample_area = []
        for i in shuffle_indices:
            if max_paste_count <= len(sample_x):
                break
            x = x_true[i]
            if bbox_true is not None:
                bbox = np.round(np.multiply(bbox_true[i], np.tile(np.shape(x)[:2][::-1], 2))).astype(int) if bbox_norm else np.array(bbox_true[i], dtype = int)
            elif mask_true is not None and 3 < np.ndim(mask_true[i]):
                bbox = instance2bbox(mask_true[i], normalize = False)
            y = np.array(y_true[i]) if y_true is not None else None
            mask = np.array(mask_true[i]) if mask_true is not None else None
            #mask = np.expand_dims(np.argmax(mask, axis = -1), axis = -1) if mask_true is not None and np.ndim(mask) == 3 and np.shape(mask)[-1] != 1 else mask
            
            h, w = np.shape(x)[:2]
            indices = np.where(np.max(0 < bbox, axis = -1, keepdims = True) != 0)[0]
            bbox = bbox[indices]
            bh, bw = (bbox[..., 3] - bbox[..., 1]), (bbox[..., 2] - bbox[..., 0])
            indices2 = np.where(np.logical_and(np.logical_and(copy_min_scale[0] <= bh, copy_min_scale[1] <= bw), 0 < (bh * bw)))[0]
            indices = indices[indices2]
            bbox = bbox[indices2]
            iou = overlap_bbox(bbox, bbox, mode = "foreground")
            indices3 = np.less(np.sum(np.greater_equal(iou, copy_iou_threshold), axis = 0), 2)
            indices = indices[indices3]
            bbox = bbox[indices3]
            if y is not None and label is not None:
                l = np.expand_dims(np.argmax(y[indices], axis = -1), axis = -1) if 1 < np.shape(y)[-1] else y[indices]
                indices4 = np.where(np.isin(l, label))[0]
                indices = indices[indices4]
                bbox = bbox[indices4]
            #sort_indices = np.arange(len(indices))
            #np.random.shuffle(sort_indices)
            #indices = indices[sort_indices]
            #bbox = bbox[sort_indices]
            for j, b in zip(indices, bbox):
                #if max_paste_count <= len(sample_x):
                #    break
                crop_x = np.array(x[b[1]:b[3], b[0]:b[2]])
                if mask is not None:
                    crop_mask = (mask[j] if 3 < np.ndim(mask) else mask)[b[1]:b[3], b[0]:b[2]]
                    region = np.greater(np.squeeze(np.expand_dims(np.argmax(crop_mask, axis = -1), axis = -1) if 2 < np.ndim(crop_mask) and 1 < np.shape(crop_mask)[-1] else crop_mask), 0.5)
                else:
                    region = np.ones([b[3] - b[1], b[2] - b[0]], dtype = bool)
                if np.sum(region) < copy_min_instance_area:
                    continue
                
                crop_x[~region] = pad_val
                sample_x.append(crop_x)
                sample_area.append(((b[3] - b[1]) * (b[2] - b[0])) / (w * h))
                if y is not None:
                    sample_y.append(y[j])
                if mask is not None:
                    sample_mask.append(crop_mask)
        
        #paste
        x_true = np.array(x_true[0])
        bbox_true = np.array(bbox_true[0]) if bbox_true is not None else None
        y_true = np.array(y_true[0]) if y_true is not None else None
        mask_true = (np.array(mask_true[0]) if 3 < np.ndim(mask_true[0]) else np.array(mask_true[0])) if mask_true is not None else None
        if bbox_true is not None:
            indices = np.where(np.max(0 < bbox_true, axis = -1, keepdims = True) != 0)[0]
            bbox_true = np.round(np.multiply(bbox_true[indices], np.tile(np.shape(x_true)[:2][::-1], 2))).astype(int) if bbox_norm else np.array(bbox_true[indices], dtype = int)
            y_true = y_true[indices] if y_true is not None else None
            mask_true = mask_true[indices] if mask_true is not None and 3 < np.ndim(mask_true) else mask_true
        #elif mask_true is not None and 3 < np.ndim(mask_true):
        #    indices = 0 < np.max(mask_true, axis = (-3, -2, -1))
        #bbox_true = bbox_true[indices] if bbox_true is not None else None
        #y_true = y_true[indices] if y_true is not None else None
        #mask_true = mask_true[indices] if mask_true is not None and 3 < np.ndim(mask_true) else mask_true
        
        if 0 < len(sample_x):
            h, w = np.shape(x_true)[:2]
            new_bbox = np.array(bbox_true) if bbox_true is not None else np.zeros((0, 4), dtype = int)
            if np.max(min_scale) < 2:
                min_scale = np.multiply(min_scale, [h, w])
            sample_indices = np.random.choice(np.arange(len(sample_x)), min(len(sample_x), max_paste_count) if not replace else max_paste_count, replace = replace)
            scales = sorted(np.random.beta(1, 1.4, size = len(sample_indices)) * np.abs(scale_range[1] - scale_range[0]) + np.min(scale_range), reverse = True)
            for i, scale in zip(sample_indices, scales):
                x = sample_x[i]
                sh, sw = np.shape(x)[:2]
                scale = (np.multiply([sh, sw], min(max([h, w]) * scale / max([sh, sw]), min([h, w]) * scale / min([sh, sw]))) + 0.5).astype(int)
                bbox = random_bbox(image_shape = [h, w], scale = scale, clip_object = clip_object, clip = False)
                pad = np.maximum([-bbox[0], -bbox[1], *np.subtract(bbox[2:], [w, h])], 0)
                bbox = np.clip(bbox, 0, [w, h, w, h])
                bh, bw = bbox[3] - bbox[1], bbox[2] - bbox[0]
                new_area = (bw * bh) / (w * h)
                
                if (min_area <= new_area, min_visibility < (new_area / (sample_area[i] + e))) and (min_scale[0] <= bh and min_scale[1] <= bw):
                    iou = overlap_bbox(new_bbox, np.expand_dims(bbox, axis = 0), mode = "foreground")
                    if np.all(np.less(iou, iou_threshold)):
                        y = sample_y[i] if i < len(sample_y) is not None else None
                        mask = sample_mask[i] if i < len(sample_mask) is not None else None
                        if sh != scale[0] or sw != scale[1]:
                            x = cv2.resize(x, scale[::-1], interpolation = method)
                            mask = cv2.resize(mask, scale[::-1], interpolation = method) if mask is not None else None
                        if np.random.random() < p_flip:
                            x = cv2.flip(x, 1)
                            mask = cv2.flip(mask, 1) if mask is not None else None
                        x = x[pad[1]:scale[0] - pad[3], pad[0]:scale[1] - pad[2]]
                        if mask is not None:
                            mask = mask[pad[1]:scale[0] - pad[3], pad[0]: scale[1] - pad[2]]
                            region = np.greater(np.squeeze(np.expand_dims(np.argmax(mask, axis = -1), axis = -1) if np.ndim(mask) == 3 and 1 < np.shape(mask)[-1] else mask), 0.5)
                        else:
                            region = np.ones([bh, bw], dtype = bool)
                        if np.sum(region) < min_instance_area:
                            continue
                        
                        x_true[bbox[1]:bbox[3], bbox[0]:bbox[2]][region] = x[region]
                        new_bbox = np.concatenate([new_bbox, np.expand_dims(bbox, axis = 0)], axis = 0)
                        if y is not None:
                            y_true = np.concatenate([y_true, np.expand_dims(y, axis = 0)])
                        if mask is not None:
                            mask = np.expand_dims(mask, axis = -1) if np.ndim(mask) < 3 else mask
                            if 3 < np.ndim(mask_true):
                                new_mask = np.zeros([h, w, np.shape(mask_true)[-1]], dtype = mask_true.dtype)
                                new_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]][region] = mask[region]
                                mask_true = np.concatenate([mask_true, np.expand_dims(new_mask, axis = 0)], axis = 0)
                            else:
                                mask_true[bbox[1]:bbox[3], bbox[0]:bbox[2]][region] = mask[region]
        if bbox_true is not None:
            bbox_true = np.divide(new_bbox, [w, h, w, h]) if bbox_norm else new_bbox
    else:
        x_true = x_true[0]
        y_true = y_true[0] if y_true is not None else None
        bbox_true = bbox_true[0] if bbox_true is not None else None
        mask_true = mask_true[0] if mask_true is not None else None
        
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def remove_background(x_true, y_true = None, bbox_true = None, mask_true = None, pad_val = 114):
    """
    x_true = (H, W, C)
    y_true(without bbox_true) = (1 or n_class)
    y_true(with bbox_true) = (P, 1 or n_class)
    bbox_true = (P, 4)
    mask_true(with bbox_true & instance mask_true) = (P, H, W, 1)
    mask_true(semantic mask_true) = (H, W, 1 or n_class)
    """
    if mask_true is not None:
        image = np.full_like(x_true, pad_val)
        for mask in (mask_true if 3 < np.ndim(mask_true) else [mask_true]):
            if 0 < np.max(mask):
                if np.ndim(mask) == 3 and np.shape(mask)[-1] != 1:
                    mask = np.expand_dims(np.argmax(mask, axis = -1), axis = -1)
                region = np.greater(mask[..., 0], 0.5)
                image[region] = x_true[region]
        x_true = image
    elif bbox_true is not None:
        image = np.full_like(x_true, pad_val)
        h, w = np.shape(x_true)[:2]
        unnorm_bbox = np.round(np.multiply(bbox_true, [w, h, w, h])).astype(int) if not np.any(np.greater_equal(bbox_true, 2)) else np.array(bbox_true, dtype = int)
        for bbox in unnorm_bbox:
            if 0 < np.max(bbox):
                image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = x_true[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        x_true = image
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result