import tensorflow as tf

from ..assign import max_iou
from ..bbox import yolo2bbox

def yolo_assign(bbox_true, bbox_pred, positive_threshold = 0.5, negative_threshold = 0.5, mode = "normal"):
    return max_iou(bbox_true, bbox_pred, positive_threshold = positive_threshold, negative_threshold = negative_threshold, mode = mode)

def yolo_target(y_true, bbox_true, score_pred, logit_pred, bbox_pred, anchors, assign = yolo_assign, sampling_count = 256, positive_ratio = 0.5, clip_ratio = 16 / 1000):
    """
    y_true = label #(padded_num_true, 1 or num_class)
    bbox_true = [[x1, y1, x2, y2], ...] #(padded_num_true, bbox)
    score_pred = classifier confidence score #(num_anchors, 1)
    logit_pred = classifier logit #(num_anchors, num_class)
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

    score_true = tf.expand_dims(tf.concat([tf.ones_like(positive_indices, dtype = tf.int32), -tf.ones_like(negative_indices, dtype = tf.int32)], axis = 0), axis = -1)
    logit_true = tf.gather(y_true, true_indices)
    bbox_true = tf.gather(bbox_true, true_indices)
    score_pred = tf.gather(score_pred, pred_indices)
    logit_pred = tf.gather(logit_pred, positive_indices)
    bbox_pred = tf.gather(bbox_pred, positive_indices)
    anchors = tf.gather(anchors, positive_indices)
    if tf.keras.backend.int_shape(true_indices)[0] != 0:
        bbox_pred = yolo2bbox(anchors, bbox_pred, clip_ratio)
    
    n_class = tf.shape(logit_true)[-1]
    negative_count = tf.shape(negative_indices)[0]
    pad_count = tf.maximum(sampling_count - tf.shape(pred_indices)[0], 0)
    score_true = tf.pad(score_true, [[0, pad_count], [0, 0]])
    logit_true = tf.cond(tf.equal(n_class, 1), true_fn = lambda: tf.pad(logit_true, [[0, negative_count + pad_count], [0, 0]]), false_fn = lambda: tf.concat([logit_true, tf.cast(tf.pad(tf.ones([negative_count + pad_count, 1]), [[0, 0], [0, n_class - 1]]), logit_true.dtype)], axis = 0))
    bbox_true = tf.pad(bbox_true, [[0, negative_count + pad_count], [0, 0]])
    score_pred = tf.pad(score_pred, [[0, pad_count], [0, 0]])
    logit_pred = tf.pad(logit_pred, [[0, negative_count + pad_count], [0, 0]])
    bbox_pred = tf.pad(bbox_pred, [[0, negative_count + pad_count], [0, 0]])
    return score_true, logit_true, bbox_true, score_pred, logit_pred, bbox_pred