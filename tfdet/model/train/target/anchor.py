import tensorflow as tf

from tfdet.core.assign import max_iou
from tfdet.core.bbox import bbox2delta

def anchor_target(y_true, bbox_true, y_pred, bbox_pred, anchors, assign = max_iou, sampling_count = 256, positive_ratio = 0.5, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2]):
    """
    y_true = label #(padded_num_true, 1 or num_class)
    bbox_true = [[x1, y1, x2, y2], ...] #(padded_num_true, bbox)
    y_pred = classifier logit #(num_anchors, num_class)
    bbox_pred = classifier regress #(num_anchors, delta)
    anchors = [[x1, y1, x2, y2], ...] #(num_anchors, bbox)
    """
    pred_count = tf.shape(anchors)[0]
    valid_indices = tf.where(tf.reduce_max(tf.cast(0 < bbox_true, tf.int32), axis = -1))
    y_true = tf.gather_nd(y_true, valid_indices)
    bbox_true = tf.gather_nd(bbox_true, valid_indices)
    
    true_indices, positive_indices, negative_indices = assign(bbox_true, anchors)
    
    if isinstance(sampling_count, int) and 0 < sampling_count:
        positive_count = tf.cast(sampling_count * positive_ratio, tf.int32)
        indices = tf.range(tf.shape(positive_indices)[0])
        indices = tf.random.shuffle(indices)[:positive_count]
        positive_indices = tf.gather(positive_indices, indices)
        true_indices = tf.gather(true_indices, indices)
        positive_count = tf.cast(tf.shape(positive_indices)[0], tf.float32)
        negative_count = tf.cast(1 / positive_ratio * positive_count - positive_count, tf.int32)
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
    else:
        sampling_count = pred_count
    pred_indices = tf.concat([positive_indices, negative_indices], axis = 0)
    
    y_true = tf.gather(y_true, true_indices)
    bbox_true = tf.gather(bbox_true, true_indices)
    y_pred = tf.gather(y_pred, pred_indices)
    bbox_pred = tf.gather(bbox_pred, positive_indices)
    anchors = tf.gather(anchors, positive_indices)
    if tf.keras.backend.int_shape(true_indices)[0] != 0:
        bbox_true = bbox2delta(bbox_true, anchors, mean, std)

    n_class = tf.shape(y_true)[-1]
    negative_count = tf.shape(negative_indices)[0]
    pad_count = tf.maximum(sampling_count - tf.shape(pred_indices)[0], 0)
    y_true = tf.cond(tf.equal(n_class, 1), true_fn = lambda: tf.pad(y_true, [[0, negative_count + pad_count], [0, 0]]), false_fn = lambda: tf.concat([y_true, tf.cast(tf.pad(tf.ones([negative_count + pad_count, 1]), [[0, 0], [0, n_class - 1]]), y_true.dtype)], axis = 0))
    bbox_true = tf.pad(bbox_true, [[0, negative_count + pad_count], [0, 0]])
    y_pred = tf.pad(y_pred, [[0, pad_count], [0, 0]])
    bbox_pred = tf.pad(bbox_pred, [[0, negative_count + pad_count], [0, 0]])
    return y_true, bbox_true, y_pred, bbox_pred