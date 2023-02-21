import functools

import tensorflow as tf

def regularize(model, weight_decay = 1e-4, loss = tf.keras.regularizers.l2):
    weight_decay = weight_decay() if callable(weight_decay) else weight_decay
    reg_loss = []
    for w in model.trainable_weights:
        if "gamma" not in w.name and "beta" not in w.name:
            l = loss(weight_decay)(w)
            reg_loss.append(l / tf.cast(tf.size(w), l.dtype))
    return reg_loss

def weight_reduce_loss(loss, weight = None, avg_factor = None):
    """
    https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/utils.py
    """
    if weight is not None:
        loss = loss * weight
    if avg_factor is not None:
        loss = tf.reduce_sum(loss) / (avg_factor + tf.keras.backend.epsilon())
    else:
        loss = tf.reduce_mean(loss)
    return loss

def resize_loss(mask_true = None, mask_pred = None, loss = None, method = "bilinear", **kwargs):
    if callable(mask_true):
        return functools.partial(resize_loss, loss = mask_true, method = method, **kwargs)
    
    mask_pred = tf.image.resize(mask_pred, tf.shape(mask_true)[-3:-1], method = method)
    return loss(mask_true, mask_pred, **kwargs)