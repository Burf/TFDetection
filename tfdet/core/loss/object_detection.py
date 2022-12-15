import tensorflow as tf
    
from ..bbox import overlap_bbox

def iou(bbox_true, bbox_pred, reduce = True, mode = "normal"):
    bbox_true = tf.reshape(bbox_true, [-1, 4])
    bbox_pred = tf.reshape(bbox_pred, [-1, 4])
    
    overlaps = overlap_bbox(bbox_pred, bbox_true, mode = mode) #(P, T)
    max_iou = tf.reduce_max(overlaps, axis = -1, keepdims = True)
    loss = 1 - max_iou
    if reduce:
        loss = tf.reduce_mean(loss)
    return loss

def giou(bbox_true, bbox_pred, reduce = True, mode = "general"):
    return iou(bbox_true, bbox_pred, reduce = reduce, mode = mode)

def ciou(bbox_true, bbox_pred, reduce = True, mode = "complete"):
    return iou(bbox_true, bbox_pred, reduce = reduce, mode = mode)

def diou(bbox_true, bbox_pred, reduce = True, mode = "distance"):
    return iou(bbox_true, bbox_pred, reduce = reduce, mode = mode)