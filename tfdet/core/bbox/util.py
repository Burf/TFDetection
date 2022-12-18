import tensorflow as tf
import numpy as np

def scale_bbox(bbox_true, scale):
    x1, y1, x2, y2 = tf.split(bbox_true, 4, axis = -1)
    w, h = (x2 - x1) / 2, (y2 - y1) / 2
    cx, cy = x1 + w, y1 + h
    w *= scale
    h *= scale
    bbox_true = tf.concat([cx - w, cy - h, cx + w, cy + h], axis = -1)
    return bbox_true

def isin(bbox_true, bbox_pred, extra_length = None, mode = "rect"):
    """
    Calculates center_pred in bbox_true
    
    extra_length = compare by center ± extra_length
    mode = ('rect', 'circle')
    """
    if mode not in ("rect", "circle"):
        raise ValueError("unknown mode '{0}'".format(mode))
        
    true_count = tf.shape(bbox_true)[0]
    pred_count = tf.shape(bbox_pred)[0]
        
    bbox_true = tf.reshape(tf.tile(tf.expand_dims(bbox_true, 0), [1, 1, pred_count]), [-1, 4])
    bbox_pred = tf.tile(bbox_pred, [true_count, 1])
    
    tx1, ty1, tx2, ty2 = tf.split(bbox_true, 4, axis = -1)
    px1, py1, px2, py2 = tf.split(bbox_pred, 4, axis = -1)
    
    tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    if extra_length is not None:
        tx1 = tcx - extra_length
        ty1 = tcy - extra_length
        tx2 = tcx + extra_length
        ty2 = tcy + extra_length
    th, tw = (ty2 - ty1) / 2, (tx2 - tx1) / 2
    
    if mode == "rect":
        #flag = 0 < (tf.maximum(tf.minimum(tx2, px2) - tf.maximum(tx1, px1), 0) * tf.maximum(tf.minimum(ty2, py2) - tf.maximum(ty1, py1), 0))
        flag = tf.logical_and(tf.logical_and(tx1 < pcx, pcx < tx2), tf.logical_and(ty1 < pcy, pcy < ty2))
    else: #elif mode == "circle":
        flag = tf.logical_and(tf.abs(tcx - pcx) < tw, tf.abs(tcy - pcy) < th)
    return tf.reshape(flag, [true_count, pred_count])

def random_bbox(alpha = 1, image_shape = None, scale = None, clip = False, clip_object = True):
    h, w = image_shape[:2] if image_shape is not None else [1, 1]
    scale_h, scale_w = [scale, scale] if np.ndim(scale) == 0 else scale
    if scale is not None and np.any(np.greater_equal(scale, 2)):
        if image_shape is None:
            h, w = [scale_h, scale_w]
        scale_h, scale_w = [scale_h / h, scale_w / w]
    elif scale is None:
        scale_h, scale_w = [np.random.beta(alpha, alpha), np.random.beta(alpha, alpha)]
    center_x, center_y = np.random.random(), np.random.random()
    if clip_object:
        center_x = center_x * (1 - scale_w) + (scale_w / 2)
        center_y = center_y * (1 - scale_h) + (scale_h / 2)
    bbox = [center_x - (scale_w / 2), center_y - (scale_h / 2), center_x + (scale_w / 2), center_y + (scale_h / 2)]
    if clip:
        bbox = np.clip(bbox, 0, 1)
    if image_shape is not None:
        bbox = np.round(np.multiply(bbox, [w, h, w, h])).astype(int)
    return bbox