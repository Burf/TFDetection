import tensorflow as tf

def scale_bbox(bbox_true, scale):
    x1, y1, x2, y2 = tf.split(bbox_true, 4, axis = -1)
    w, h = (x2 - x1) / 2, (y2 - y1) / 2
    cx, cy = x1 + w, y1 + h
    w *= scale
    h *= scale
    bbox_true = tf.concat([cx - w, cy - h, cx + w, cy + h], axis = -1)
    bbox_true = tf.clip_by_value(bbox_true, 0, 1)
    return bbox_true

def isin(bbox_true, bbox_pred):
    true_count = tf.shape(bbox_true)[0]
    pred_count = tf.shape(bbox_pred)[0]
        
    bbox_true = tf.reshape(tf.tile(tf.expand_dims(bbox_true, 0), [1, 1, pred_count]), [-1, 4])
    bbox_pred = tf.tile(bbox_pred, [true_count, 1])
    
    tx1, ty1, tx2, ty2 = tf.split(bbox_true, 4, axis = -1)
    px1, py1, px2, py2 = tf.split(bbox_pred, 4, axis = -1)
    
    pcx, pcy = (px2 - px1) / 2, (py2 - py1) / 2
    
    flag = tf.logical_and(tf.logical_and(tx1 < pcx, pcx < tx2), tf.logical_and(ty1 < pcy, pcy < ty2)) 
    flag = tf.reshape(flag, [true_count, pred_count])
    return flag