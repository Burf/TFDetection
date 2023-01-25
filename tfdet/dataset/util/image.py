import cv2
import numpy as np

def load_image(path, bgr2rgb = True):
    image = path
    if isinstance(path, str):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if bgr2rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def save_image(image, path, rgb2bgr = True):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if rgb2bgr else image
    image = cv2.imwrite(path, image)
    return path

def instance2semantic(y_true, mask_true, threshold = 0.5, ignore_label = 0, label = None):
    """
    y_true = y_true #(pad_num, 1)
    mask_true = instance mask #(pad_num, h, w, 1)
    
    semantic_true = semantic mask #(h, w, 1)
    """
    mask_true = np.array(mask_true, dtype = np.uint8) if not isinstance(mask_true, np.ndarray) else mask_true
    if 3 < np.ndim(mask_true):
        y_true = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
        if not np.issubdtype(y_true.dtype, np.number):
            if label is None:
                raise ValueError("if y_true is string, label is required.")
            else:
                if 0 < len(y_true):
                    y_true = np.expand_dims(np.argmax(y_true == label, axis = -1).astype(np.int32), axis = -1)
                else:
                    y_true = np.zeros_like(y_true, dtype = np.int32)
        h, w = np.shape(mask_true)[1:3]
        valid_indices = np.argwhere(y_true != ignore_label)[:, 0]
        y_true = y_true[valid_indices]
        mask_true = mask_true[valid_indices]
        region = threshold < mask_true
        area = np.sum(region, axis = (1, 2))
        valid_indices = np.argwhere(0 < area)[:, 0]
        y_true = y_true[valid_indices]
        region = region[valid_indices]
        area = area[valid_indices]
        sort_indices = np.argsort(-area[:, 0])
        y_true = y_true[sort_indices]
        region = region[sort_indices]
        semantic_true = np.full([h, w, 1], ignore_label, dtype = np.uint16)
        for y, r in zip(y_true, region):
            semantic_true[r] = y[0]
        mask_true = semantic_true
    return mask_true

def instance2bbox(mask_true, normalize = False, threshold = 0.5):
    """
    mask_true = instance mask #(pad_num, h, w, 1)
    """
    mask_true = np.array(mask_true, dtype = np.uint8) if not isinstance(mask_true, np.ndarray) else mask_true
    if 3 < np.ndim(mask_true):
        padded_num_true, h, w = np.shape(mask_true)[:3]

        bbox_true = []
        for index in range(padded_num_true):
            pos = np.where(np.greater(mask_true[index], threshold))[:2]
            if 0 < len(pos[0]):
                bbox = [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]) + 1, np.max(pos[0]) + 1] #x1, y1, x2, y2
                bbox = np.clip(bbox, 0, [w, h, w, h])
            else:
                bbox = [0, 0, 0, 0]
            bbox_true.append(bbox)

        bbox_true = (np.divide(bbox_true, [w, h, w, h], dtype = np.float32) if normalize else np.array(bbox_true, dtype = np.int32)) if 0 < padded_num_true else np.zeros((0, 4), dtype = np.int32)
        return bbox_true
    else:
        return np.zeros((0, 4), dtype = mask_true.dtype)

def instance2panoptic(y_true, mask_true, threshold = 0.5, divisor = 1000, ignore_label = 0, label = None):
    """
    y_true = y_true #(pad_num, 1)
    mask_true = instance mask #(pad_num, h, w, 1)
    
    panoptic_true = panoptic mask #(h, w, 1)
    """
    mask_true = np.array(mask_true, dtype = np.uint8) if not isinstance(mask_true, np.ndarray) else mask_true
    if 3 < np.ndim(mask_true):
        y_true = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
        if not np.issubdtype(y_true.dtype, np.number):
            if label is None:
                raise ValueError("if y_true is string, label is required.")
            else:
                if 0 < len(y_true):
                    y_true = np.expand_dims(np.argmax(y_true == label, axis = -1).astype(np.int32), axis = -1)
                else:
                    y_true = np.zeros_like(y_true, dtype = np.int32)
        h, w = np.shape(mask_true)[1:3]
        valid_indices = np.argwhere(y_true != ignore_label)[:, 0]
        y_true = y_true[valid_indices]
        mask_true = mask_true[valid_indices]
        region = threshold < mask_true
        area = np.sum(region, axis = (1, 2))
        valid_indices = np.argwhere(0 < area)[:, 0]
        y_true = y_true[valid_indices]
        region = region[valid_indices]
        area = area[valid_indices]
        sort_indices = np.argsort(-area[:, 0])
        y_true = y_true[sort_indices]
        region = region[sort_indices]

        unique_id = np.unique(y_true)
        ignore_flag = unique_id == ignore_label
        if np.any(ignore_flag):
            unique_id = unique_id[~ignore_flag]

        store = {k:0 for k in unique_id}
        panoptic_true = np.full([h, w, 1], ignore_label, dtype = np.uint32)
        for y, r in zip(y_true, region):
            cls_id = y[0]
            new_id = cls_id * divisor + store[cls_id]
            panoptic_true[r] = new_id
            store[cls_id] += 1
        mask_true = panoptic_true
    return mask_true

def panoptic2instance(y_true, mask_true, divisor = 1000, ignore_label = 0, label = None):
    """
    y_true = y_true #(pad_num, 1)
    mask_true = panoptic mask #(h, w, 1)
    
    instance_true = panoptic mask #(pad_num, h, w, 1)
    """
    mask_true = np.array(mask_true, dtype = np.uint32) if not isinstance(mask_true, np.ndarray) else mask_true
    if np.ndim(mask_true) < 4:
        y_true = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
        if not np.issubdtype(y_true.dtype, np.number):
            if label is None:
                raise ValueError("if y_true is string, label is required.")
            else:
                if 0 < len(y_true):
                    y_true = np.expand_dims(np.argmax(y_true == label, axis = -1).astype(np.int32), axis = -1)
                else:
                    y_true = np.zeros_like(y_true, dtype = np.int32)
        #valid_indices = np.argwhere(y_true != ignore_label)[:, 0]
        #y_true = y_true[valid_indices]
        
        h, w = np.shape(mask_true)[:2]
        store = {k:0 for k in np.unique(y_true)}
        instance_true = np.full([len(y_true), h, w, 1], ignore_label, dtype = np.uint8)
        for i, y in enumerate(y_true):
            cls_id = y[0]
            if cls_id != ignore_label:
                new_id = cls_id * divisor + store[cls_id]
                instance_true[i, mask_true == new_id] = 1
                store[cls_id] += 1
        mask_true = instance_true
    return mask_true

def trim_bbox(x_true, image_shape = None, pad_val = 0, mode = "both", decimal = 4):
    """
    mode = ("left", "right", "both")
    """
    if mode not in ("left", "right", "both"):
        raise ValueError("unknown mode '{0}'".format(mode))
    h, w = np.shape(x_true)[:2]
    if image_shape is not None:
        l = r = [0, 0]
        p = [max([h - image_shape[0], 0]), max(w - image_shape[1], 0)]
        if mode == "left":
            l = p
        elif mode == "right":
            r = p
        elif mode == "both":
            l = np.divide(p, 2, dtype = np.int32)
            r = np.subtract(p, l)
        bbox = [l[1], l[0], w - r[1], h - r[0]]
    else:
        image = np.round(x_true, decimal)
        bbox = [0, 0, w, h]
        while 0 < np.shape(image)[1]: #left
            if np.mean(image[:, 0]) != pad_val:
                break
            image = image[:, 1:]
            bbox[0] += 1
        while 0 < np.shape(image)[0]: #top
            if np.mean(image[0]) != pad_val:
                break
            image = image[1:]
            bbox[1] += 1
        while 0 < np.shape(image)[1]: #right
            if np.mean(image[:, -1]) != pad_val:
                break
            image = image[:, :-1]
            bbox[2] -= 1
        while 0 < np.shape(image)[0]: #bottom
            if np.mean(image[-1]) != pad_val:
                break
            image = image[:-1]
            bbox[3] -= 1
    if (bbox[3] - bbox[1]) == 0 or (bbox[2] - bbox[0]) == 0:
        bbox = [0, 0, w, h]
    return bbox
