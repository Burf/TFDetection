import tensorflow as tf

from ..bbox import overlap_bbox, isin
from ..util import euclidean_matrix

def atss(bbox_true, bbox_pred, k = 9, threshold = 0.01, min_threshold = 0.0001, mode = "normal"):
    #https://arxiv.org/abs/1912.02424
    k = tf.minimum(k, tf.shape(bbox_pred)[0])
    overlaps = overlap_bbox(bbox_true, bbox_pred, mode = mode) #(T, P)
    dist = euclidean_matrix(bbox_true, bbox_pred) #(T, P)
    sort_indices = tf.argsort(dist, axis = -1) #(T, P)
    candidate_indices = sort_indices[..., :k] #(T, K)
    
    candidate_overlaps = tf.gather(overlaps, candidate_indices, batch_dims = -1) #(T, K)
    candidate_threshold = tf.reduce_mean(candidate_overlaps, axis = -1) + tf.math.reduce_std(candidate_overlaps, axis = -1)
    candidate_flag = tf.greater_equal(candidate_overlaps, tf.expand_dims(candidate_threshold, axis = -1))
    
    true_count = tf.shape(bbox_true)[0]
    pred_count = tf.shape(bbox_pred)[0]
    candidate_flag = tf.concat([candidate_flag, tf.zeros((true_count, pred_count - k), dtype = tf.bool)], axis = -1) #(T, K) + (T, P - K)
    candidate_flag = tf.gather(candidate_flag, tf.argsort(sort_indices, axis = -1), batch_dims = -1) #(T, P)
    
    isin_flag = isin(bbox_true, bbox_pred) #(T, P)
    overlaps = tf.where(tf.logical_and(candidate_flag, isin_flag), overlaps, 0) #(T, P)
    overlaps = tf.transpose(overlaps) #(P, T)
    
    max_iou = tf.reduce_max(overlaps, axis = -1)
    match = tf.where(max(threshold, min_threshold) <= max_iou, 1, -1)
    
    positive_indices = tf.where(match == 1)[:, 0]
    negative_indices = tf.where(match == -1)[:, 0]
    
    positive_overlaps = tf.gather(overlaps, positive_indices)
    true_indices = tf.cond(tf.greater(tf.shape(positive_overlaps)[1], 0), true_fn = lambda: tf.argmax(positive_overlaps, axis = -1), false_fn = lambda: tf.cast(tf.constant([]), tf.int64))
    return true_indices, positive_indices, negative_indices