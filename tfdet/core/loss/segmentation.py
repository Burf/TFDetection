import tensorflow as tf

from .cross_entropy import binary_cross_entropy

def dice(y_true, y_pred, smooth = 1., weight = None, reduce = True):
    n_true_class = tf.shape(y_true)[-1]
    n_pred_class = tf.shape(y_pred)[-1]
    
    y_true = tf.cast(tf.cond(tf.logical_and(tf.equal(n_true_class, 1), tf.not_equal(n_pred_class, 1)), true_fn = lambda: tf.one_hot(tf.cast(y_true, tf.int32), n_pred_class)[:, 0], false_fn = lambda: y_true), y_pred.dtype)
    y_pred = y_pred / (tf.reduce_sum(y_pred, axis = -1, keepdims = True) + tf.keras.backend.epsilon())
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
    
    axis = tf.range(tf.rank(y_true))[1:-1]
    inter = tf.reduce_sum(y_true * y_pred, axis = axis)
    total = tf.reduce_sum(y_true + y_pred, axis = axis)
    score = (2 * inter + smooth + tf.keras.backend.epsilon()) / (total + smooth + tf.keras.backend.epsilon())
    loss = 1 - score
    if weight is not None:
        loss *= weight
    if reduce:
        loss = tf.reduce_sum(loss, axis = -1)
        loss = tf.reduce_mean(loss)
    return loss

def bce_dice(y_true, y_pred, smooth = 1., weight = None, reduce = True):
    bce_loss = binary_cross_entropy(y_true, y_pred, focal = False, reduce = False)
    dice_loss = dice(y_true, y_pred, smooth, reduce = False)
    bce_loss = tf.reduce_sum(bce_loss, axis = tf.range(tf.rank(bce_loss))[1:-1])
    loss = bce_loss + dice_loss
    if weight is not None:
        loss *= weight
    if reduce:
        loss = tf.reduce_sum(loss, axis = -1)
        loss = tf.reduce_mean(loss)
    return loss

def tversky(y_true, y_pred, smooth = 1., alpha = 0.5, beta = 0.5, weight = None, reduce = True):
    n_true_class = tf.shape(y_true)[-1]
    n_pred_class = tf.shape(y_pred)[-1]
    
    y_true = tf.cast(tf.cond(tf.logical_and(tf.equal(n_true_class, 1), tf.not_equal(n_pred_class, 1)), true_fn = lambda: tf.one_hot(tf.cast(y_true, tf.int32), n_pred_class)[:, 0], false_fn = lambda: y_true), y_pred.dtype)
    y_pred = y_pred / (tf.reduce_sum(y_pred, axis = -1, keepdims = True) + tf.keras.backend.epsilon())
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
    
    axis = tf.range(tf.rank(y_true))[1:-1]
    TP = tf.reduce_sum(y_true * y_pred, axis = axis)
    FP = tf.reduce_sum(y_true * (1 - y_pred), axis = axis)
    FN = tf.reduce_sum((1 - y_true) * y_pred, axis = axis)

    score = (TP + smooth + tf.keras.backend.epsilon()) / (TP + alpha * FP + beta * FN + smooth + tf.keras.backend.epsilon())
    loss = 1 - score
    if weight is not None:
        loss *= weight
    if reduce:
        loss = tf.reduce_sum(loss, axis = -1)
        loss = tf.reduce_mean(loss)
    return loss

def focal_tversky(y_true, y_pred, smooth = 1., alpha = 0.5, beta = 0.5, gamma = 1., weight = None, reduce = True):
    loss = tversky(y_true, y_pred, smooth = smooth, alpha = alpha, beta = beta, reduce = False)
    loss = tf.pow(loss, gamma)
    if weight is not None:
        loss *= weight
    if reduce:
        loss = tf.reduce_sum(loss, axis = -1)
        loss = tf.reduce_mean(loss)
    return loss

def iou_pixcel(y_true, y_pred, smooth = 1., weight = None, reduce = True):
    n_true_class = tf.shape(y_true)[-1]
    n_pred_class = tf.shape(y_pred)[-1]
    
    y_true = tf.cast(tf.cond(tf.logical_and(tf.equal(n_true_class, 1), tf.not_equal(n_pred_class, 1)), true_fn = lambda: tf.one_hot(tf.cast(y_true, tf.int32), n_pred_class)[:, 0], false_fn = lambda: y_true), y_pred.dtype)
    y_pred = y_pred / (tf.reduce_sum(y_pred, axis = -1, keepdims = True) + tf.keras.backend.epsilon())
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
    
    axis = tf.range(tf.rank(y_true))[1:-1]
    inter = tf.reduce_sum(y_true * y_pred, axis = axis)
    total = tf.reduce_sum(y_true + y_pred, axis = axis)
    union = total - inter

    iou = (inter + smooth + tf.keras.backend.epsilon()) / (union + smooth + tf.keras.backend.epsilon())
    loss = 1 - iou
    if weight is not None:
        loss *= weight
    if reduce:
        loss = tf.reduce_sum(loss, axis = -1)
        loss = tf.reduce_mean(loss)
    return loss

def generalized_dice(y_true, y_pred, smooth = 1, weight = None, reduce = True):
    n_true_class = tf.shape(y_true)[-1]
    n_pred_class = tf.shape(y_pred)[-1]
    
    y_true = tf.cast(tf.cond(tf.logical_and(tf.equal(n_true_class, 1), tf.not_equal(n_pred_class, 1)), true_fn = lambda: tf.one_hot(tf.cast(y_true, tf.int32), n_pred_class)[:, 0], false_fn = lambda: y_true), y_pred.dtype)
    y_pred = y_pred / (tf.reduce_sum(y_pred, axis = -1, keepdims = True) + tf.keras.backend.epsilon())
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
    
    axis = tf.range(tf.rank(y_true))[1:-1]
    w = 1 / (tf.pow(tf.reduce_sum(y_true, axis = axis), 2) + tf.keras.backend.epsilon())
    w = tf.clip_by_value(w, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
    numerator = w * tf.reduce_sum(y_true * y_pred, axis = axis)
    denominator = w * tf.reduce_sum(y_true + y_pred, axis = axis)
    
    score = (2 * numerator + smooth + tf.keras.backend.epsilon()) / (denominator + smooth + tf.keras.backend.epsilon())
    loss = 1 - score
    if weight is not None:
        loss *= weight
    if reduce:
        loss = tf.reduce_sum(loss, axis = -1)
        loss = tf.reduce_mean(loss)
    return loss

def bce_generalized_dice(y_true, y_pred, smooth = 1, weight = None, reduce = True):
    bce_loss = binary_cross_entropy(y_true, y_pred, focal = False, reduce = False)
    dice_loss = generalized_dice(y_true, y_pred, smooth = smooth, reduce = False)
    bce_loss = tf.reduce_sum(bce_loss, axis = tf.range(tf.rank(bce_loss))[1:-1])
    loss = bce_loss + dice_loss
    if weight is not None:
        loss *= weight
    if reduce:
        loss = tf.reduce_sum(loss, axis = -1)
        loss = tf.reduce_mean(loss)
    return loss