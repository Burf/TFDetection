import tensorflow as tf

from ..bbox import overlap_bbox

def max_iou(bbox_true, bbox_pred, positive_threshold = 0.5, negative_threshold = 0.4, mode = "normal"):
    overlaps = tf.transpose(overlap_bbox(bbox_true, bbox_pred, mode = mode)) #(P, T)
    max_iou = tf.reduce_max(overlaps, axis = -1)

    match = tf.where(max_iou < negative_threshold, -1, 0)
    match = tf.where(tf.logical_and(positive_threshold <= max_iou, 0 < max_iou), 1, match)
    
    positive_indices = tf.where(match == 1)[:, 0]
    negative_indices = tf.where(match == -1)[:, 0]
    
    positive_overlaps = tf.gather(overlaps, positive_indices)
    true_indices = tf.cond(tf.greater(tf.shape(positive_overlaps)[1], 0), true_fn = lambda: tf.argmax(positive_overlaps, axis = -1), false_fn = lambda: tf.cast(tf.constant([]), tf.int64))
    return true_indices, positive_indices, negative_indices