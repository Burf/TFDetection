import tensorflow as tf

from tfdet.core.assign import point
from tfdet.core.loss import regularize as regularize_loss
from tfdet.core.util import map_fn
from .loss.fcos import classnet_accuracy, classnet_loss, boxnet_loss, centernessnet_loss
from .target import fcos_target

def train_model(input, logits, regress, points, centerness = None,
                assign = point, sampling_count = 256, positive_ratio = 0.5,
                batch_size = 1, regularize = True, weight_decay = 1e-4, focal = True, alpha = .25, gamma = 2., sigma = 3, class_weight = None, missing_value = 0.):
    if not isinstance(logits, list):
        logits, regress, points = [logits], [regress], [points]
        if centerness is not None:
            centerness = [centerness]
    
    y_true = tf.keras.layers.Input(shape = (None, None), name = "y_true", dtype = logits.dtype)
    bbox_true = tf.keras.layers.Input(shape = (None, 4), name = "bbox_true", dtype = regress[0].dtype)
    
    #[[0, 64], [64, 128], [128, 256], [256, 512], [512, float("inf")]] > [[0, 0.0625], [0.0625, 0.125], [0.125, 0.25], [0.25, 0.5], [0.5, 1.0]]
    regress_range = [0.] + [1 / (2 ** (index - 1)) for index in range(len(points), 0, -1)]
    regress_range = [[regress_range[index], regress_range[index + 1]] for index in range(len(points))]
    regress_range = [tf.ones_like(p) * r for p, r in zip(points, regress_range)]
    
    logits = tf.concat(logits, axis = -2)
    regress = tf.concat(regress, axis = -2)
    points = tf.concat(points, axis = 0)
    regress_range = tf.concat(regress_range, axis = 0)
    if centerness is not None:
        centerness = tf.concat(centerness, axis = -2)
        
    points = tf.tile(tf.expand_dims(points, axis = 0), [tf.shape(input)[0], 1, 1])
    regress_range = tf.tile(tf.expand_dims(regress_range, axis = 0), [tf.shape(input)[0], 1, 1])
    
    if centerness is not None:
        target_y_true, target_bbox_true, target_centerness_true, target_y_pred, target_bbox_pred, target_centerness_pred = tf.keras.layers.Lambda(lambda args: map_fn(fcos_target, *args, dtype = (y_true.dtype, bbox_true.dtype, bbox_true.dtype, logits.dtype, regress.dtype, centerness.dtype), batch_size = batch_size, assign = assign, sampling_count = sampling_count, positive_ratio = positive_ratio), name = "fcos_target")([y_true, bbox_true, logits, regress, points, regress_range, centerness])
    else:
        target_y_true, target_bbox_true, target_y_pred, target_bbox_pred = tf.keras.layers.Lambda(lambda args: map_fn(fcos_target, *args, dtype = (y_true.dtype, bbox_true.dtype, logits.dtype, regress.dtype), batch_size = batch_size, assign = assign, sampling_count = sampling_count, positive_ratio = positive_ratio), name = "fcos_target")([y_true, bbox_true, logits, regress, points, regress_range])
    
    score_accuracy = tf.keras.layers.Lambda(lambda args: classnet_accuracy(*args, missing_value = missing_value), name = "score_accuracy")([target_y_true, target_y_pred])
    score_loss = tf.keras.layers.Lambda(lambda args: classnet_loss(*args, focal = focal, alpha = alpha, gamma = gamma, weight = class_weight, missing_value = missing_value), name = "score_loss")([target_y_true, target_y_pred])
    regress_loss = tf.keras.layers.Lambda(lambda args: boxnet_loss(*args, sigma = sigma, missing_value = missing_value), name = "regress_loss")([target_y_true, target_bbox_true, target_bbox_pred])
    loss = {"score_loss":score_loss, "regress_loss":regress_loss}
    if centerness is not None:
        centerness_loss = tf.keras.layers.Lambda(lambda args: centernessnet_loss(*args, missing_value = missing_value), name = "centerness_loss")([target_y_true, target_centerness_true, target_centerness_pred])
        loss["centerness_loss"] = centerness_loss
    
    score_accuracy = tf.expand_dims(score_accuracy, axis = -1)
    loss = {k:tf.expand_dims(v, axis = -1) for k, v in loss.items()}

    model = tf.keras.Model([input, y_true, bbox_true], tf.reduce_sum(list(loss.values()), keepdims = True, name = "loss"))
    
    model.add_metric(score_accuracy, name = "score_accuracy", aggregation = "mean")
    for k, v in loss.items():
        model.add_metric(v, name = k, aggregation = "mean")
    
    for k, v in loss.items():
        model.add_loss(v)

    if regularize:
        reg_loss = regularize_loss(model, weight_decay)
        model.add_loss(lambda: tf.reduce_sum(reg_loss, keepdims = True, name = "regularize_loss"))
    return model