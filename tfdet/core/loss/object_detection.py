import tensorflow as tf
    
from ..bbox import iou as iou_calculator

def iou(bbox_true, bbox_pred, reduce = tf.reduce_mean, mode = "normal"):
    iou = iou_calculator(bbox_pred, bbox_true, mode = mode)
    loss = 1 - iou
    if reduce:
        loss = reduce(loss)
    return loss

def giou(bbox_true, bbox_pred, reduce = tf.reduce_mean, mode = "general"):
    return iou(bbox_true, bbox_pred, reduce = reduce, mode = mode)

def ciou(bbox_true, bbox_pred, reduce = tf.reduce_mean, mode = "complete"):
    return iou(bbox_true, bbox_pred, reduce = reduce, mode = mode)

def diou(bbox_true, bbox_pred, reduce = tf.reduce_mean, mode = "distance"):
    return iou(bbox_true, bbox_pred, reduce = reduce, mode = mode)