import tensorflow as tf

from tfdet.core.assign import max_iou
from tfdet.core.loss import regularize as regularize_loss
from tfdet.core.util import map_fn
from .loss.retina import classnet_accuracy, classnet_loss, boxnet_loss
from .target import anchor_target

def train_model(input, logits, regress, anchors,
                assign = max_iou, sampling_count = 256, positive_ratio = 0.5,
                batch_size = 1, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], regularize = True, weight_decay = 1e-4, focal = True, alpha = .25, gamma = 1.5, sigma = 3, class_weight = None, missing_value = 0.):
    y_true = tf.keras.layers.Input(shape = (None, None), name = "y_true", dtype = logits.dtype)
    bbox_true = tf.keras.layers.Input(shape = (None, 4), name = "bbox_true", dtype = regress.dtype)
    
    anchors = tf.tile(tf.expand_dims(anchors, axis = 0), [tf.shape(input)[0], 1, 1])
    target_y_true, target_bbox_true, target_y_pred, target_bbox_pred = tf.keras.layers.Lambda(lambda args: map_fn(anchor_target, *args, dtype = (y_true.dtype, bbox_true.dtype, logits.dtype, regress.dtype), batch_size = batch_size, 
                                                                                                                  assign = assign, sampling_count = sampling_count, positive_ratio = positive_ratio, mean = mean,std = std), name = "anchor_target")([y_true, bbox_true, logits, regress, anchors])
    
    score_accuracy = tf.keras.layers.Lambda(lambda args: classnet_accuracy(*args, missing_value = missing_value), name = "score_accuracy")([target_y_true, target_y_pred])
    score_loss = tf.keras.layers.Lambda(lambda args: classnet_loss(*args, focal = focal, alpha = alpha, gamma = gamma, weight = class_weight, missing_value = missing_value), name = "score_loss")([target_y_true, target_y_pred])
    regress_loss = tf.keras.layers.Lambda(lambda args: boxnet_loss(*args, sigma = sigma, missing_value = missing_value), name = "regress_loss")([target_y_true, target_bbox_true, target_bbox_pred])
    
    score_accuracy = tf.expand_dims(score_accuracy, axis = -1)
    score_loss = tf.expand_dims(score_loss, axis = -1)
    regress_loss = tf.expand_dims(regress_loss, axis = -1)
    
    model = tf.keras.Model([input, y_true, bbox_true], tf.reduce_sum([score_loss, regress_loss], keepdims = True, name = "loss"))
    
    model.add_metric(score_accuracy, name = "score_accuracy", aggregation = "mean")
    model.add_metric(score_loss, name = "score_loss", aggregation = "mean")
    model.add_metric(regress_loss, name = "regress_loss", aggregation = "mean")
    model.add_loss(score_loss)
    model.add_loss(regress_loss)

    if regularize:
        reg_loss = regularize_loss(model, weight_decay)
        model.add_loss(lambda: tf.reduce_sum(reg_loss, keepdims = True, name = "regularize_loss"))
    return model