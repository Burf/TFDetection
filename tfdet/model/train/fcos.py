import tensorflow as tf

from tfdet.core.assign import point
from tfdet.core.loss import binary_cross_entropy, focal_binary_cross_entropy, iou, regularize as regularize_loss
from .loss import AnchorFreeLoss
from ..postprocess.anchor_free import FilterDetection

def focal_loss(y_true, y_pred, alpha = .25, gamma = 2., weight = None, reduce = tf.reduce_mean):
    return focal_binary_cross_entropy(y_true, y_pred, alpha = alpha, gamma = gamma, weight = weight, reduce = reduce)

def train_model(input, y_pred, bbox_pred, points, conf_pred = None,
                assign = point, sampler = None,
                proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05, soft_nms = False, ignore_label = 0, performance_count = 5000,
                class_loss = focal_loss, bbox_loss = iou, conf_loss = binary_cross_entropy,
                regularize = True, weight_decay = 1e-4,
                decode_bbox = True, class_weight = None, background = False, 
                batch_size = 1, missing_value = 0.):
    if not isinstance(y_pred, (tuple, list)):
        y_pred = [y_pred]
    if not isinstance(bbox_pred, (tuple, list)):
        bbox_pred = [bbox_pred]
    if not isinstance(points, (tuple, list)):
        points = [points]
    if conf_pred is not None and not isinstance(conf_pred, (tuple, list)):
        conf_pred = [conf_pred]
    
    y_true = tf.keras.layers.Input(shape = (None, None), name = "y_true")
    bbox_true = tf.keras.layers.Input(shape = (None, 4), name = "bbox_true")
    
    #[[0, 64], [64, 128], [128, 256], [256, 512], [512, float("inf")]] > [[0, 0.0625], [0.0625, 0.125], [0.125, 0.25], [0.25, 0.5], [0.5, 1.0]]
    regress_range = [0.] + [1 / (2 ** (index - 1)) for index in range(len(points), 0, -1)]
    regress_range = [[regress_range[index], regress_range[index + 1]] for index in range(len(points))]
    regress_range = [tf.ones_like(p) * r for p, r in zip(points, regress_range)]
    
    args = [arg for arg in [y_pred, bbox_pred, points, regress_range, conf_pred] if arg is not None]
    out = AnchorFreeLoss(class_loss = class_loss, bbox_loss = bbox_loss, conf_loss = conf_loss,
                         decode_bbox = decode_bbox, weight = class_weight, background = background,
                         assign = assign, sampler = sampler,
                         batch_size = batch_size,
                         missing_value = missing_value, dtype = tf.float32, name = "anchor_free_loss")([y_true, bbox_true], args)
    args = [arg for arg in [y_pred, bbox_pred, points, conf_pred] if arg is not None]
    y_pred, bbox_pred = FilterDetection(proposal_count = proposal_count, iou_threshold = iou_threshold, score_threshold = score_threshold, soft_nms = soft_nms, ignore_label = ignore_label, performance_count = performance_count,
                                        batch_size = batch_size, dtype = tf.float32, name = "filter_detection")(args)
    model = tf.keras.Model([input, y_true, bbox_true], [y_pred, bbox_pred])
    
    loss_class, loss_bbox = out[:2]
    loss_conf = out[2] if 2 < len(out) else None
    loss_class = [loss_class] if not isinstance(loss_class, (tuple, list)) else loss_class
    loss_bbox = [loss_bbox] if not isinstance(loss_bbox, (tuple, list)) else loss_bbox
    loss_conf = [loss_conf] if loss_conf is not None and not isinstance(loss_conf, (tuple, list)) else loss_conf
    #for level, (_loss_class, _loss_bbox) in enumerate(zip(loss_class, loss_bbox)):
    #    model.add_loss(tf.expand_dims(_loss_class, axis = -1))
    #    model.add_loss(tf.expand_dims(_loss_bbox, axis = -1))
    #    if loss_conf is not None:
    #        model.add_loss(tf.expand_dims(loss_conf[level], axis = -1))
    
    losses = []
    loss_class = tf.expand_dims(tf.reduce_sum(loss_class), axis = -1)
    loss_bbox = tf.expand_dims(tf.reduce_sum(loss_bbox), axis = -1)
    model.add_metric(loss_class, name = "loss_class", aggregation = "mean")
    model.add_metric(loss_bbox, name = "loss_bbox", aggregation = "mean")
    losses += [loss_class, loss_bbox]
    if loss_conf is not None:
        loss_conf = tf.expand_dims(tf.reduce_sum(loss_conf), axis = -1)
        model.add_metric(loss_conf, name = "loss_conf", aggregation = "mean")
        losses += [loss_conf]
        
    losses = tf.reduce_sum(losses, axis = 0)
    model.add_loss(losses)
    
    if regularize:
        model.add_loss(lambda: tf.cast(tf.reduce_sum(regularize_loss(model, weight_decay), keepdims = True), tf.float32))
    return model