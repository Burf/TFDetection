import tensorflow as tf

from ..bbox import overlap_point

def point(y_true, bbox_true, y_pred, point_pred, regress_range = None, threshold = 0.0001, min_threshold = 0.0001):
    overlaps = tf.transpose(overlap_point(bbox_true, point_pred, regress_range)) #(P, T)
    max_area = tf.reduce_max(overlaps, axis = -1)
    match = tf.where(max(threshold, min_threshold) <= max_area, 1, -1)
    
    positive_indices = tf.where(match == 1)[:, 0]
    negative_indices = tf.where(match == -1)[:, 0]
    
    positive_overlaps = tf.gather(overlaps, positive_indices)
    true_indices = tf.cond(tf.greater(tf.shape(positive_overlaps)[1], 0), true_fn = lambda: tf.argmax(positive_overlaps, axis = -1), false_fn = lambda: tf.cast(tf.constant([]), tf.int64))
    return true_indices, positive_indices, negative_indices