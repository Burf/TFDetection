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

def iou(bbox_true, bbox_pred, mode = "normal", e = 1e-12):
    if mode not in ("normal", "foreground", "general", "complete", "distance"):
        raise ValueError("unknown mode '{0}'".format(mode))
    
    tx1, ty1, tx2, ty2 = tf.split(bbox_true, 4, axis = -1)
    px1, py1, px2, py2 = tf.split(bbox_pred, 4, axis = -1)
    
    area_true = (tx2 - tx1) * (ty2 - ty1)
    area_pred = (px2 - px1) * (py2 - py1)
    
    x1 = tf.maximum(tx1, px1)
    y1 = tf.maximum(ty1, py1)
    x2 = tf.minimum(tx2, px2)
    y2 = tf.minimum(ty2, py2)
    
    inter = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    if mode != "foreground":
        union = area_true + area_pred - inter
    else:
        union = area_true
    iou = tf.clip_by_value(inter / (union + tf.keras.backend.epsilon()), 0, 1)
    if mode in ["general", "complete", "distance"]:
        gx1 = tf.minimum(tx1, px1)
        gy1 = tf.minimum(ty1, py1)
        gx2 = tf.maximum(tx2, px2)
        gy2 = tf.maximum(ty2, py2)
        
        if mode == "general":
            general_inter = tf.maximum(gx2 - gx1, 0) * tf.maximum(gy2 - gy1, 0)
            iou = giou = tf.clip_by_value(iou - (general_inter - union) / (general_inter + tf.keras.backend.epsilon()), 0, 1)
        else:
            tw = tf.maximum(tx2 - tx1, 0)
            th = tf.maximum(ty2 - ty1, 0)
            pw = tf.maximum(px2 - px1, 0)
            ph = tf.maximum(py2 - py1, 0)
            ctx = tx1 + 0.5 * tw
            cty = ty1 + 0.5 * th
            cpx = px1 + 0.5 * pw
            cpy = py1 + 0.5 * ph
            c = (gx2 - gx1) ** 2 + (gy2 - gy1) ** 2
            rho = (ctx - cpx) ** 2 + (cty - cpy) ** 2
            
            diou = tf.clip_by_value(iou - rho / (c + tf.keras.backend.epsilon()), 0, 1)
            if mode == "distance": 
                iou = diou
            else: #complete
                v = ((tf.math.atan(pw / (ph + tf.keras.backend.epsilon()))
                    - tf.math.atan(tw / (th + tf.keras.backend.epsilon()))) * 2 / np.pi) ** 2
                alpha = v / (1 - iou + v + tf.keras.backend.epsilon())
                iou = ciou = tf.clip_by_value(diou - alpha * v, 0, 1)
    return iou

def iou_numpy(bbox_true, bbox_pred, mode = "normal", e = 1e-12):
    if mode not in ("normal", "foreground", "general", "complete", "distance"):
        raise ValueError("unknown mode '{0}'".format(mode))
    
    tx1, ty1, tx2, ty2 = np.split(bbox_true, 4, axis = -1)
    px1, py1, px2, py2 = np.split(bbox_pred, 4, axis = -1)
    
    area_true = (tx2 - tx1) * (ty2 - ty1)
    area_pred = (px2 - px1) * (py2 - py1)
    
    x1 = np.maximum(tx1, px1)
    y1 = np.maximum(ty1, py1)
    x2 = np.minimum(tx2, px2)
    y2 = np.minimum(ty2, py2)
    
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    if mode != "foreground":
        union = area_true + area_pred - inter
    else:
        union = area_true
    iou = np.clip(inter / (union + e), 0, 1)
    if mode in ["general", "complete", "distance"]:
        gx1 = np.minimum(tx1, px1)
        gy1 = np.minimum(ty1, py1)
        gx2 = np.maximum(tx2, px2)
        gy2 = np.maximum(ty2, py2)
        
        if mode == "general":
            general_inter = np.maximum(gx2 - gx1, 0) * np.maximum(gy2 - gy1, 0)
            iou = giou = np.clip(iou - (general_inter - union) / (general_inter + e), 0, 1)
        else:
            tw = np.maximum(tx2 - tx1, 0)
            th = np.maximum(ty2 - ty1, 0)
            pw = np.maximum(px2 - px1, 0)
            ph = np.maximum(py2 - py1, 0)
            ctx = tx1 + 0.5 * tw
            cty = ty1 + 0.5 * th
            cpx = px1 + 0.5 * pw
            cpy = py1 + 0.5 * ph
            c = (gx2 - gx1) ** 2 + (gy2 - gy1) ** 2
            rho = (ctx - cpx) ** 2 + (cty - cpy) ** 2
            
            diou = np.clip(iou - rho / (c + e), 0, 1)
            if mode == "distance": 
                iou = diou
            else: #complete
                atan = np.vectorize(np.math.atan)
                v = ((atan(pw / (ph + e))
                    - atan(tw / (th + e))) * 2 / np.pi) ** 2
                alpha = v / (1 - iou + v + e)
                iou = ciou = np.clip(diou - alpha * v, 0, 1)
    return iou

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
        bbox = np.round(np.multiply(bbox, [w, h, w, h])).astype(np.int32)
    return bbox