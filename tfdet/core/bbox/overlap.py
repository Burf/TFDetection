import tensorflow as tf
import numpy as np

from .util import iou, iou_numpy

def overlap_bbox(bbox_true, bbox_pred, mode = "normal"):
    """
    bbox_true = [[x1, y1, x2, y2], ...] #(N, bbox)
    bbox_pred = [[x1, y1, x2, y2], ...] #(M, bbox)
    
    overlaps = true & pred iou matrix #(N, M)
    """
    if mode not in ("normal", "foreground", "general", "complete", "distance"):
        raise ValueError("unknown mode '{0}'".format(mode))
    
    true_count = tf.shape(bbox_true)[0]
    pred_count = tf.shape(bbox_pred)[0]
        
    bbox_true = tf.reshape(tf.tile(tf.expand_dims(bbox_true, 0), [1, 1, pred_count]), [-1, 4])
    bbox_pred = tf.tile(bbox_pred, [true_count, 1])
    
    overlaps = iou(bbox_true, bbox_pred, mode = mode)
    overlaps = tf.reshape(overlaps, [true_count, pred_count])
    return overlaps

def overlap_point(bbox_true, points, regress_range = None):
    true_count = tf.shape(bbox_true)[0]
    pred_count = tf.shape(points)[0]

    bbox_true = tf.reshape(tf.tile(tf.expand_dims(bbox_true, 0), [1, 1, pred_count]), [-1, 4])
    points = tf.tile(points, [true_count, 1])
    
    x1, y1, x2, y2 = tf.split(bbox_true, 4, axis = -1)
    px, py = tf.split(points, 2, axis = -1)
    area = tf.reshape((x2 - x1) * (y2 - y1), [true_count, pred_count])
    offset = tf.concat([px - x1, py - y1, x2 - px, y2 - py], axis = -1) #left, top, right, bottom
    offset = tf.reshape(offset, [true_count, pred_count, 4])
    min_offset = tf.reduce_min(offset, axis = -1)
    
    overlap_flag = tf.greater(min_offset, 0)
    if regress_range is not None:
        max_offset = tf.reduce_max(offset, axis = -1)
        regress_range = tf.tile(regress_range, [true_count, 1])
        regress_range = tf.reshape(regress_range, [true_count, pred_count, 2])
        range_flag = tf.logical_and(tf.greater(max_offset, regress_range[..., 0]), tf.less_equal(max_offset, regress_range[..., 1]))
        overlap_flag = tf.logical_and(overlap_flag, range_flag)
    pad_area = tf.where(overlap_flag, area, tf.reduce_max(area) + 1)
    min_flag = tf.equal(area, tf.reduce_min(pad_area, axis = 0, keepdims = True))
    overlaps = tf.where(min_flag, area, 0)
    return overlaps

def overlap_bbox_numpy(bbox_true, bbox_pred, mode = "normal", e = 1e-12):
    """
    bbox_true = [[x1, y1, x2, y2], ...] #(N, bbox)
    bbox_pred = [[x1, y1, x2, y2], ...] #(M, bbox)
    
    overlaps = true & pred iou matrix #(N, M)
    """
    if mode not in ("normal", "foreground", "general", "complete", "distance"):
        raise ValueError("unknown mode '{0}'".format(mode))
    
    true_count = np.shape(bbox_true)[0]
    pred_count = np.shape(bbox_pred)[0]
        
    bbox_true = np.reshape(np.tile(np.expand_dims(bbox_true, 0), [1, 1, pred_count]), [-1, 4])
    bbox_pred = np.tile(bbox_pred, [true_count, 1])
    
    overlaps = iou_numpy(bbox_true, bbox_pred, mode = mode, e = e)
    overlaps = np.reshape(overlaps, [true_count, pred_count])
    return overlaps