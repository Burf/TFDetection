import tensorflow as tf

def bbox2delta(bbox_true, bbox_pred, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2]):
    true_h = bbox_true[..., 3] - bbox_true[..., 1]
    true_w = bbox_true[..., 2] - bbox_true[..., 0]
    true_center_x = bbox_true[..., 0] + 0.5 * true_w
    true_center_y = bbox_true[..., 1] + 0.5 * true_h
    
    pred_h = bbox_pred[..., 3] - bbox_pred[..., 1]
    pred_w = bbox_pred[..., 2] - bbox_pred[..., 0]
    pred_h = tf.where(pred_h <= 0, tf.keras.backend.epsilon(), pred_h)
    pred_w = tf.where(pred_w <= 0, tf.keras.backend.epsilon(), pred_w)
    pred_center_x = bbox_pred[..., 0] + 0.5 * pred_w
    pred_center_y = bbox_pred[..., 1] + 0.5 * pred_h
    
    x = (true_center_x - pred_center_x) / pred_w
    y = (true_center_y - pred_center_y) / pred_h
    w = true_w / pred_w
    h = true_h / pred_h
    w = tf.math.log(tf.where(w <= 0, tf.keras.backend.epsilon(), w))
    h = tf.math.log(tf.where(h <= 0, tf.keras.backend.epsilon(), h))

    delta = tf.stack([x, y, w, h], axis = -1)
    if mean is not None and std is not None:
        delta = (delta - mean) / (tf.cast(std, delta.dtype) + tf.keras.backend.epsilon())
    return delta

def delta2bbox(bbox, delta, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000):
    if mean is not None and std is not None:
        delta = delta * std + mean
    h = bbox[..., 3] - bbox[..., 1]
    w = bbox[..., 2] - bbox[..., 0]
    center_x = bbox[..., 0] + 0.5 * w
    center_y = bbox[..., 1] + 0.5 * h
    center_x += delta[..., 0] * w
    center_y += delta[..., 1] * h
    delta_h = delta[..., 3]
    delta_w = delta[..., 2]
    if isinstance(clip_ratio, float):
        clip_value = tf.math.abs(tf.math.log(clip_ratio))
        delta_h = tf.clip_by_value(delta_h, -clip_value, clip_value)
        delta_w = tf.clip_by_value(delta_w, -clip_value, clip_value)
    h *= tf.exp(delta_h)
    w *= tf.exp(delta_w)
    x1 = center_x - 0.5 * w
    y1 = center_y - 0.5 * h
    x2 = x1 + w
    y2 = y1 + h
    bbox = tf.stack([x1, y1, x2, y2], axis = -1)
    return bbox

def bbox2offset(bbox_true, points):
    x1, y1, x2, y2 = tf.split(bbox_true, 4, axis = -1)
    px, py = tf.split(points, 2, axis = -1)
    offset = tf.concat([px - x1, py - y1, x2 - px, y2 - py], axis = -1) #left, top, right, bottom
    return offset

def offset2bbox(points, offset):
    x, y = tf.split(points, 2, axis = -1)
    left, top, right, bottom = tf.split(offset, 4, axis = -1)
    bbox = tf.concat([x - left, y -top, x + right, y + bottom], axis = -1)
    return bbox

def offset2centerness(offset):
    left, top, right, bottom = tf.split(offset, 4, axis = -1)
    lr = tf.concat([left, right], axis = -1)
    tb = tf.concat([top, bottom], axis = -1)
    max_lr = tf.reduce_max(lr, axis = -1, keepdims = True)
    max_tb = tf.reduce_max(tb, axis = -1, keepdims = True)
    centerness = tf.sqrt((tf.reduce_min(lr, axis = -1, keepdims = True) / tf.where(max_lr == 0, tf.keras.backend.epsilon(), max_lr)) * (tf.reduce_min(tb, axis = -1, keepdims = True) / tf.where(max_tb == 0, tf.keras.backend.epsilon(), max_tb)))
    return centerness