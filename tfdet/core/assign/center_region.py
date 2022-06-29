import tensorflow as tf

from ..bbox import overlap_bbox, scale_bbox, isin

def center_region(bbox_true, bbox_pred, positive_scale = 0.2, negative_scale = 0.5, threshold = 0.01, min_threshold = 0.0001, mode = "normal"):
    #https://arxiv.org/abs/1901.03278
    pos_bbox_true = scale_bbox(bbox_true, positive_scale)
    neg_bbox_true = scale_bbox(bbox_true, negative_scale)
    
    pos_flag = tf.transpose(isin(pos_bbox_true, bbox_pred)) #(P, T)
    neg_flag = tf.transpose(~isin(neg_bbox_true, bbox_pred)) #(P, T)
    #ignore_flag = tf.logical_and(~pos_flag, ~neg_flag)
    overlaps = tf.transpose(overlap_bbox(bbox_true, bbox_pred, mode = mode)) #(P, T)
    overlaps = tf.where(pos_flag, overlaps, 0)
    neg_overlaps = tf.where(neg_flag, -1, 0)
    
    max_iou = tf.reduce_max(overlaps, axis = -1)
    match = tf.reduce_min(neg_overlaps, axis = -1)
    match = tf.where(max(threshold, min_threshold) <= max_iou, 1, match)
    
    positive_indices = tf.where(match == 1)[:, 0]
    negative_indices = tf.where(match == -1)[:, 0]
    
    positive_overlaps = tf.gather(overlaps, positive_indices)
    true_indices = tf.cond(tf.greater(tf.shape(positive_overlaps)[1], 0), true_fn = lambda: tf.argmax(positive_overlaps, axis = -1), false_fn = lambda: tf.cast(tf.constant([]), tf.int64))
    return true_indices, positive_indices, negative_indices