import cv2
import numpy as np

def load_image(path, bgr2rgb = True):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED) if isinstance(path, str) else path
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if bgr2rgb else image
    return image

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