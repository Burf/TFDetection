import tensorflow as tf

from tfdet.core.assign import point
from tfdet.core.loss import binary_cross_entropy, focal_binary_cross_entropy, iou, regularize as regularize_loss
from tfdet.core.util import map_fn
from .loss.fcos import classnet_accuracy, classnet_loss, boxnet_loss, centernessnet_loss
from .target import fcos_target
from ..postprocess.fcos import FilterDetection

def focal_loss(y_true, y_pred, alpha = .25, gamma = 2., weight = None, reduce = tf.reduce_mean):
    return focal_binary_cross_entropy(y_true, y_pred, alpha = alpha, gamma = gamma, weight = weight, reduce = reduce)

def train_model(input, logits, regress, points, centerness = None,
                assign = point, sampling_count = None, positive_ratio = 0.5,
                proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05, soft_nms = False, ignore_label = 0, performance_count = 5000,
                batch_size = 1, 
                class_loss = focal_loss, bbox_loss = iou, centerness_loss = binary_cross_entropy, regularize = True, weight_decay = 1e-4, 
                class_weight = None, background = False, missing_value = 0.):
    if not isinstance(logits, list):
        logits, regress, points = [logits], [regress], [points]
        if centerness is not None:
            centerness = [centerness]
    
    y_true = tf.keras.layers.Input(shape = (None, None), name = "y_true")
    bbox_true = tf.keras.layers.Input(shape = (None, 4), name = "bbox_true")
    
    #[[0, 64], [64, 128], [128, 256], [256, 512], [512, float("inf")]] > [[0, 0.0625], [0.0625, 0.125], [0.125, 0.25], [0.25, 0.5], [0.5, 1.0]]
    regress_range = [0.] + [1 / (2 ** (index - 1)) for index in range(len(points), 0, -1)]
    regress_range = [[regress_range[index], regress_range[index + 1]] for index in range(len(points))]
    regress_range = [tf.ones_like(p) * r for p, r in zip(points, regress_range)]
    
    concat_logits = tf.concat(logits, axis = -2)
    concat_regress = tf.concat(regress, axis = -2)
    concat_points = tf.concat(points, axis = 0)
    regress_range = tf.concat(regress_range, axis = 0)
    if centerness is not None:
        concat_centerness = tf.concat(centerness, axis = -2)
        
    tile_points = tf.tile(tf.expand_dims(concat_points, axis = 0), [tf.shape(input)[0], 1, 1])
    regress_range = tf.tile(tf.expand_dims(regress_range, axis = 0), [tf.shape(input)[0], 1, 1])
    
    if centerness is not None:
        target_y_true, target_bbox_true, target_centerness_true, target_y_pred, target_bbox_pred, target_centerness_pred = tf.keras.layers.Lambda(lambda args: map_fn(fcos_target, *args, dtype = (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32), batch_size = batch_size, assign = assign, sampling_count = sampling_count, positive_ratio = positive_ratio), dtype = tf.float32, name = "fcos_target")([y_true, bbox_true, concat_logits, concat_regress, tile_points, regress_range, concat_centerness])
    else:
        target_y_true, target_bbox_true, target_y_pred, target_bbox_pred = tf.keras.layers.Lambda(lambda args: map_fn(fcos_target, *args, dtype = (tf.float32, tf.float32, tf.float32, tf.float32), batch_size = batch_size, assign = assign, sampling_count = sampling_count, positive_ratio = positive_ratio), dtype = tf.float32, name = "fcos_target")([y_true, bbox_true, concat_logits, concat_regress, tile_points, regress_range])
    
    score_accuracy = tf.keras.layers.Lambda(lambda args: classnet_accuracy(*args, missing_value = missing_value), dtype = tf.float32, name = "score_accuracy")([target_y_true, target_y_pred])
    score_loss = tf.keras.layers.Lambda(lambda args: classnet_loss(*args, loss = class_loss, weight = class_weight, background = background, missing_value = missing_value), dtype = tf.float32, name = "score_loss")([target_y_true, target_y_pred])
    regress_loss = tf.keras.layers.Lambda(lambda args: boxnet_loss(*args, loss = bbox_loss, missing_value = missing_value), dtype = tf.float32, name = "regress_loss")([target_y_true, target_bbox_true, target_bbox_pred])
    loss = {"score_loss":score_loss, "regress_loss":regress_loss}
    if centerness is not None:
        _centerness_loss = tf.keras.layers.Lambda(lambda args: centernessnet_loss(*args, loss = centerness_loss, missing_value = missing_value), dtype = tf.float32, name = "centerness_loss")([target_y_true, target_centerness_true, target_centerness_pred])
        loss["centerness_loss"] = _centerness_loss
    
    score_accuracy = tf.expand_dims(score_accuracy, axis = -1)
    loss = {k:tf.expand_dims(v, axis = -1) for k, v in loss.items()}

    
    y_pred, bbox_pred = FilterDetection(proposal_count = proposal_count, iou_threshold = iou_threshold, score_threshold = score_threshold, soft_nms = soft_nms, ignore_label = ignore_label, performance_count = performance_count,
                                        batch_size = batch_size, dtype = tf.float32)([l for l in [logits, regress, points, centerness] if l is not None])
    model = tf.keras.Model([input, y_true, bbox_true], [y_pred, bbox_pred])
    
    model.add_metric(score_accuracy, name = "score_accuracy", aggregation = "mean")
    for k, v in loss.items():
        model.add_metric(v, name = k, aggregation = "mean")
    
    for k, v in loss.items():
        model.add_loss(v)

    if regularize:
        model.add_loss(lambda: tf.cast(tf.reduce_sum(regularize_loss(model, weight_decay), keepdims = True, name = "regularize_loss"), tf.float32))
    return model