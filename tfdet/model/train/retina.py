import functools

import tensorflow as tf

from tfdet.core.assign import max_iou
from tfdet.core.loss import focal_binary_cross_entropy, smooth_l1, regularize as regularize_loss
from tfdet.core.util import map_fn
from .loss.retina import classnet_accuracy, classnet_loss, boxnet_loss
from .target import anchor_target
from ..postprocess.retina import FilterDetection

def focal_loss(y_true, y_pred, alpha = .25, gamma = 1.5, weight = None, reduce = True):
    return focal_binary_cross_entropy(y_true, y_pred, alpha = alpha, gamma = gamma, weight = weight, reduce = reduce)

def train_model(input, logits, regress, anchors,
                assign = max_iou, sampling_count = None, positive_ratio = 0.5,
                proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05, soft_nms = False, valid = False, ignore_label = 0, performance_count = 5000,
                batch_size = 1, mean = [0., 0., 0., 0.], std = [1., 1., 1., 1.], clip_ratio = 16 / 1000,
                class_loss = focal_loss, bbox_loss = smooth_l1, regularize = True, weight_decay = 1e-4, 
                class_weight = None, background = False, missing_value = 0.):
    y_true = tf.keras.layers.Input(shape = (None, None), name = "y_true")
    bbox_true = tf.keras.layers.Input(shape = (None, 4), name = "bbox_true")
    
    tile_anchors = tf.tile(tf.expand_dims(anchors, axis = 0), [tf.shape(input)[0], 1, 1])
    target_y_true, target_bbox_true, target_y_pred, target_bbox_pred = tf.keras.layers.Lambda(lambda args: map_fn(anchor_target, *args, dtype = (tf.float32, tf.float32, tf.float32, tf.float32), batch_size = batch_size, 
                                                                                                                  assign = assign, sampling_count = sampling_count, positive_ratio = positive_ratio, valid = valid, mean = mean, std = std), dtype = tf.float32, name = "anchor_target")([y_true, bbox_true, logits, regress, tile_anchors])
    
    score_accuracy = tf.keras.layers.Lambda(lambda args: classnet_accuracy(*args, missing_value = missing_value), dtype = tf.float32, name = "score_accuracy")([target_y_true, target_y_pred])
    score_loss = tf.keras.layers.Lambda(lambda args: classnet_loss(*args, loss = class_loss, weight = class_weight, background = background, missing_value = missing_value), dtype = tf.float32, name = "score_loss")([target_y_true, target_y_pred])
    regress_loss = tf.keras.layers.Lambda(lambda args: boxnet_loss(*args, loss = bbox_loss, missing_value = missing_value), dtype = tf.float32, name = "regress_loss")([target_y_true, target_bbox_true, target_bbox_pred])
    
    score_accuracy = tf.expand_dims(score_accuracy, axis = -1)
    score_loss = tf.expand_dims(score_loss, axis = -1)
    regress_loss = tf.expand_dims(regress_loss, axis = -1)
    
    y_pred, bbox_pred = FilterDetection(proposal_count = proposal_count, iou_threshold = iou_threshold, score_threshold = score_threshold, soft_nms = soft_nms, valid = valid, ignore_label = ignore_label, performance_count = performance_count,
                                        batch_size = batch_size, mean = mean, std = std, clip_ratio = clip_ratio, dtype = tf.float32)([logits, regress, anchors])
    model = tf.keras.Model([input, y_true, bbox_true], [y_pred, bbox_pred])
    
    model.add_metric(score_accuracy, name = "score_accuracy", aggregation = "mean")
    model.add_metric(score_loss, name = "score_loss", aggregation = "mean")
    model.add_metric(regress_loss, name = "regress_loss", aggregation = "mean")
    model.add_loss(score_loss)
    model.add_loss(regress_loss)

    if regularize:
        model.add_loss(lambda: tf.cast(tf.reduce_sum(regularize_loss(model, weight_decay), keepdims = True, name = "regularize_loss"), tf.float32))
    return model