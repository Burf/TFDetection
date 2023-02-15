import tensorflow as tf

from tfdet.core.assign import max_iou
from tfdet.core.loss import binary_cross_entropy, focal_binary_cross_entropy, ciou, regularize as regularize_loss
from .loss import YoloLoss
from ..postprocess.yolo import FilterDetection

def focal_loss(y_true, y_pred, alpha = .25, gamma = 2., weight = None, reduce = tf.reduce_mean):
    return focal_binary_cross_entropy(y_true, y_pred, alpha = alpha, gamma = gamma, weight = weight, reduce = reduce)

def train_model(input, score_pred, logit_pred, bbox_pred, anchors,
                assign = max_iou, sampler = None,
                proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05, soft_nms = False, valid_inside_anchor = False, ignore_label = 0, performance_count = 5000,
                clip_ratio = 16 / 1000,
                score_loss = binary_cross_entropy, class_loss = focal_loss, bbox_loss = ciou,
                regularize = True, weight_decay = 1e-4, 
                decode_bbox = True, class_weight = None, 
                batch_size = 1, missing_value = 0.):
    y_true = tf.keras.layers.Input(shape = (None, None), name = "y_true")
    bbox_true = tf.keras.layers.Input(shape = (None, 4), name = "bbox_true")
    
    loss_score, loss_class, loss_bbox = YoloLoss(score_loss = score_loss, class_loss = class_loss, bbox_loss = bbox_loss,
                                                 decode_bbox = decode_bbox, valid_inside_anchor = valid_inside_anchor, weight = class_weight,
                                                 assign = assign, sampler = sampler,
                                                 clip_ratio = clip_ratio,
                                                 batch_size = batch_size,
                                                 missing_value = missing_value, dtype = tf.float32, name = "yolo_loss")([y_true, bbox_true], [score_pred, logit_pred, bbox_pred, anchors])
    y_pred, bbox_pred = FilterDetection(proposal_count = proposal_count, iou_threshold = iou_threshold, score_threshold = score_threshold, soft_nms = soft_nms, valid_inside_anchor = valid_inside_anchor, ignore_label = ignore_label, performance_count = performance_count,
                                        clip_ratio = clip_ratio,
                                        batch_size = batch_size, dtype = tf.float32, name = "filter_detection")([score_pred, logit_pred, bbox_pred, anchors])
    model = tf.keras.Model([input, y_true, bbox_true], [y_pred, bbox_pred])
    
    loss_score = [loss_score] if not isinstance(loss_score, (tuple, list)) else loss_score
    loss_class = [loss_class] if not isinstance(loss_class, (tuple, list)) else loss_class
    loss_bbox = [loss_bbox] if not isinstance(loss_bbox, (tuple, list)) else loss_bbox
    #for _loss_score, _loss_class, _loss_bbox in zip(loss_score, loss_class, loss_bbox):
    #    model.add_loss(tf.expand_dims(_loss_score, axis = -1))
    #    model.add_loss(tf.expand_dims(_loss_class, axis = -1))
    #    model.add_loss(tf.expand_dims(_loss_bbox, axis = -1))
        
    loss_score = tf.expand_dims(tf.reduce_sum(loss_score), axis = -1)
    loss_class = tf.expand_dims(tf.reduce_sum(loss_class), axis = -1)
    loss_bbox = tf.expand_dims(tf.reduce_sum(loss_bbox), axis = -1)
    model.add_metric(loss_score, name = "loss_score", aggregation = "mean")
    model.add_metric(loss_class, name = "loss_class", aggregation = "mean")
    model.add_metric(loss_bbox, name = "loss_bbox", aggregation = "mean")
    
    losses = tf.reduce_sum([loss_score, loss_class, loss_bbox], axis = 0)
    model.add_loss(losses)
    
    if regularize:
        model.add_loss(lambda: tf.cast(tf.reduce_sum(regularize_loss(model, weight_decay), keepdims = True), tf.float32))
    return model