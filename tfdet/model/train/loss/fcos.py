import tensorflow as tf

from tfdet.core.loss import binary_cross_entropy, focal_binary_cross_entropy, iou
from .retina import classnet_accuracy, classnet_loss as _classnet_loss, boxnet_loss as _boxnet_loss

def classnet_loss(y_true, y_pred, loss = focal_binary_cross_entropy, weight = None, background = False, missing_value = 0.):
    return _classnet_loss(y_true, y_pred, loss = loss, weight = weight, background = background, missing_value = missing_value)
    
def boxnet_loss(y_true, bbox_true, bbox_pred, loss = iou, missing_value = 0.):
    return _boxnet_loss(y_true, bbox_true, bbox_pred, loss = loss, missing_value = missing_value)

def centernessnet_loss(y_true, centerness_true, centerness_pred, loss = binary_cross_entropy, missing_value = 0.):
    """
    y_true = targeted y_true #(batch_size, sampling_count, 1 or num_class)
    centerness_true = targeted centerness_true #(batch_size, sampling_count, 1)
    centerness_pred = targeted centerenss_pred #(batch_size, sampling_count, 1)
    """
    y_true = tf.cond(tf.equal(tf.shape(y_true)[-1], 1), true_fn = lambda: y_true, false_fn = lambda: tf.expand_dims(tf.cast(tf.argmax(y_true, -1), y_true.dtype), axis = -1))
    y_true = tf.reshape(y_true, (-1,))
    centerness_true = tf.reshape(centerness_true, (-1, 1))
    centerness_pred = tf.reshape(centerness_pred, (-1, 1))
    
    true_indices = tf.where(0 < y_true)[:, 0]
    centerness_true = tf.gather(centerness_true, true_indices)
    centerness_pred = tf.gather(centerness_pred, true_indices)
    
    _loss = loss(centerness_true, centerness_pred, reduce = False)
    _loss = tf.reduce_mean(_loss)
    _loss = tf.where(tf.math.is_nan(_loss), missing_value, _loss)
    return _loss