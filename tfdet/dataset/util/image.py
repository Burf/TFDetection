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

def instance2semantic(y_true, mask_true, label = None):
    """
    y_true = (P, 1)
    mask_true = (P, H, W, 1)
    """
    if 3 < np.ndim(mask_true):
        area = np.sum(mask_true != 0, axis = (1, 2))[..., 0]
        sort_indices = np.argsort(-area)
        h, w = np.shape(mask_true)[1:3]
        new_mask = np.zeros((h, w, 1))
        for y, mask in zip(y_true[sort_indices][..., 0], mask_true[sort_indices]):
            if label is not None and isinstance(y, str):
                y = np.argmax(np.array(label) == y)
            flag = mask != 0
            new_mask[flag] = (mask[flag] * y)
        mask_true = new_mask
    return mask_true

def instance2bbox(mask_true, normalize = False):
    #mask_true = #(padded_num_true, h, w, 1)
    padded_num_true, h, w = np.shape(mask_true)[:3]

    bbox_true = []
    for index in range(padded_num_true):
        mask = mask_true[index]
        if 0 < np.max(mask):
            pos = np.where(np.greater(mask, 0.5))[:2]
            bbox = [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]) + 1, np.max(pos[0]) + 1] #x1, y1, x2, y2
            bbox = np.clip(bbox, 0, [w, h, w, h])
        else:
            bbox = [0, 0, 0, 0]
        bbox_true.append(bbox)
    
    bbox_true = (np.divide(bbox_true, [w, h, w, h]).astype(np.float32) if normalize else np.array(bbox_true, dtype = int)) if 0 < padded_num_true else np.zeros((0, 4), dtype = int)
    return bbox_true

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
            l = np.divide(p, 2).astype(int)
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