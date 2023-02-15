import tensorflow as tf

from tfdet.core.assign import max_iou
from tfdet.core.loss import focal_binary_cross_entropy, smooth_l1, regularize as regularize_loss
from .loss import AnchorLoss
from ..postprocess.anchor import FilterDetection

def train_model(input, y_pred, bbox_pred, anchors,
                assign = max_iou, sampler = None, valid_inside_anchor = False,
                proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05, soft_nms = False, ignore_label = 0, performance_count = 5000,
                mean = [0., 0., 0., 0.], std = [1., 1., 1., 1.], clip_ratio = 16 / 1000,
                class_loss = focal_binary_cross_entropy, bbox_loss = smooth_l1,
                regularize = True, weight_decay = 1e-4, 
                decode_bbox = False, class_weight = None, background = False, 
                batch_size = 1, missing_value = 0.):
    y_true = tf.keras.layers.Input(shape = (None, None), name = "y_true")
    bbox_true = tf.keras.layers.Input(shape = (None, 4), name = "bbox_true")
    
    loss_class, loss_bbox = AnchorLoss(class_loss = class_loss, bbox_loss = bbox_loss,
                                       decode_bbox = decode_bbox, valid_inside_anchor = valid_inside_anchor, weight = class_weight, background = background,
                                       assign = assign, sampler = sampler,
                                       mean = mean, std = std, clip_ratio = clip_ratio,
                                       batch_size = batch_size,
                                       missing_value = missing_value, dtype = tf.float32, name = "anchor_loss")([y_true, bbox_true], [y_pred, bbox_pred, anchors])
    y_pred, bbox_pred = FilterDetection(proposal_count = proposal_count, iou_threshold = iou_threshold, score_threshold = score_threshold, soft_nms = soft_nms, valid_inside_anchor = valid_inside_anchor, ignore_label = ignore_label, performance_count = performance_count,
                                        mean = mean, std = std, clip_ratio = clip_ratio,
                                        batch_size = batch_size, dtype = tf.float32, name = "filter_detection")([y_pred, bbox_pred, anchors])
    model = tf.keras.Model([input, y_true, bbox_true], [y_pred, bbox_pred])
    
    loss_class = [loss_class] if not isinstance(loss_class, (tuple, list)) else loss_class
    loss_bbox = [loss_bbox] if not isinstance(loss_bbox, (tuple, list)) else loss_bbox
    #for _loss_class, _loss_bbox in zip(loss_class, loss_bbox):
    #    model.add_loss(tf.expand_dims(_loss_class, axis = -1))
    #    model.add_loss(tf.expand_dims(_loss_bbox, axis = -1))
        
    loss_class = tf.expand_dims(tf.reduce_sum(loss_class), axis = -1)
    loss_bbox = tf.expand_dims(tf.reduce_sum(loss_bbox), axis = -1)
    model.add_metric(loss_class, name = "loss_class", aggregation = "mean")
    model.add_metric(loss_bbox, name = "loss_bbox", aggregation = "mean")
    
    losses = tf.reduce_sum([loss_class, loss_bbox], axis = 0)
    model.add_loss(losses)
    
    if regularize:
        model.add_loss(lambda: tf.cast(tf.reduce_sum(regularize_loss(model, weight_decay), keepdims = True), tf.float32))
    return model