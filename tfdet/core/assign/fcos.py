import tensorflow as tf

from ..bbox import overlap_fcos

def fcos(bbox_true, point_pred, regress_range, threshold = 0.0001):
    overlaps = tf.transpose(overlap_fcos(bbox_true, point_pred, regress_range)) #(P, T)
    max_area = tf.reduce_max(overlaps, axis = -1)
    match = tf.where(tf.logical_and(threshold <= max_area, 0 < max_area), 1, -1)
    
    positive_indices = tf.where(match == 1)[:, 0]
    negative_indices = tf.where(match == -1)[:, 0]
    
    positive_overlaps = tf.gather(overlaps, positive_indices)
    true_indices = tf.cond(tf.greater(tf.shape(positive_overlaps)[1], 0), true_fn = lambda: tf.argmax(positive_overlaps, axis = -1), false_fn = lambda: tf.cast(tf.constant([]), tf.int64))
    return true_indices, positive_indices, negative_indices