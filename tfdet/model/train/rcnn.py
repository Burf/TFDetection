import functools

import tensorflow as tf
import numpy as np

from tfdet.core.assign import max_iou, random_sampler
from tfdet.core.loss import binary_cross_entropy, categorical_cross_entropy, focal_categorical_cross_entropy, smooth_l1, regularize as regularize_loss
from tfdet.core.util import map_fn
from .loss import AnchorLoss, RoiBboxLoss, RoiMaskLoss, FusedSemanticLoss
from ..postprocess.roi import FilterDetection

def smooth_l1_sigma1(y_true, y_pred, sigma = 1, reduce = tf.reduce_mean):
    return smooth_l1(y_true, y_pred, sigma = sigma, reduce = reduce)

def rpn_assign(y_true, bbox_true, y_pred, bbox_pred, positive_threshold = 0.7, negative_threshold = 0.3, min_threshold = 0.3, match_low_quality = True, mode = "normal"):
    return max_iou(y_true, bbox_true, y_pred, bbox_pred, positive_threshold = positive_threshold, negative_threshold = negative_threshold, min_threshold = min_threshold, match_low_quality = match_low_quality, mode = mode)

def train_model(input, rpn_y_pred, rpn_bbox_pred, anchors, cls_y_pred, cls_bbox_pred, proposals, cls_mask_pred = None, semantic_pred = None, train_tag = None,
                rpn_assign = rpn_assign, rpn_sampler = random_sampler,
                proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05, soft_nms = False, ensemble = True, valid_inside_anchor = True, ignore_label = 0, performance_count = 5000,
                rpn_mean = [0., 0., 0., 0.], rpn_std = [1., 1., 1., 1.], rpn_clip_ratio = 16 / 1000, 
                cls_mean = [0., 0., 0., 0.], cls_std = [0.1, 0.1, 0.2, 0.2], cls_clip_ratio = 16 / 1000,
                method = "bilinear",
                rpn_class_loss = binary_cross_entropy, rpn_bbox_loss = smooth_l1,
                cls_class_loss = focal_categorical_cross_entropy, cls_bbox_loss = smooth_l1_sigma1,
                mask_loss = binary_cross_entropy, semantic_loss = categorical_cross_entropy,
                regularize = True, weight_decay = 1e-4,
                decode_bbox = False, class_weight = None, stage_weight = [1.0, 0.5, 0.25], semantic_weight = 0.2,
                batch_size = 1, missing_value = 0.):
    if isinstance(cls_mask_pred, dict):
        train_tag = cls_mask_pred
        cls_mask_pred = None
    elif isinstance(semantic_pred, dict):
        train_tag = semantic_pred
        semantic_pred = None
    if cls_mask_pred is not None and not isinstance(cls_mask_pred, (tuple, list)) and tf.keras.backend.ndim(cls_mask_pred) == 4:
        semantic_pred = cls_mask_pred
        cls_mask_pred = None
    if train_tag is None:
        raise ValueError("please check train argument during model initialization.\nExample:\n> out = tfdet.model.detector.faster_rcnn(train = True, ...)\n> model = tfdet.model.train.rcnn.train_model(x, *out, ...)")
    
    out = train_tag["inputs"]
    y_true, bbox_true = out[:2]
    mask_true = None
    if 2 < len(out):
        mask_true = out[2]
    
    loss_rpn_class, loss_rpn_bbox = AnchorLoss(class_loss = rpn_class_loss, bbox_loss = rpn_bbox_loss,
                                               decode_bbox = decode_bbox, valid_inside_anchor = valid_inside_anchor, background = True,
                                               assign = rpn_assign, sampler = rpn_sampler,
                                               mean = rpn_mean, std = rpn_std, clip_ratio = rpn_clip_ratio,
                                               batch_size = batch_size,
                                               missing_value = missing_value, dtype = tf.float32, name = "rpn_loss")([y_true, bbox_true], [rpn_y_pred, rpn_bbox_pred, anchors])
        
    n_stage = len(train_tag["bbox_inputs"])
    sampling = train_tag["sampling"]
    loss_roi_class_list, loss_roi_bbox_list, loss_roi_mask_list = [], [], []
    for stage, (bbox_inputs, bbox_outputs) in enumerate(zip(train_tag["bbox_inputs"], train_tag["bbox_outputs"])):
        loss_roi_class, loss_roi_bbox = RoiBboxLoss(class_loss = cls_class_loss, bbox_loss = cls_bbox_loss,
                                                    sampling = sampling, decode_bbox = decode_bbox, weight = class_weight, background = True,
                                                    mean = cls_mean, std = np.divide(cls_std, stage + 1), clip_ratio = cls_clip_ratio,
                                                    missing_value = missing_value, dtype = tf.float32, name = "roi_bbox_loss_{0}".format(stage + 1) if 1 < n_stage else "roi_bbox_loss")(bbox_inputs, bbox_outputs)
        loss_roi_class_list.append(loss_roi_class * stage_weight[stage])
        loss_roi_bbox_list.append(loss_roi_bbox * stage_weight[stage])
        
        if 0 < len(train_tag["mask_inputs"]):
            loss_roi_mask = RoiMaskLoss(loss = mask_loss, missing_value = missing_value, dtype = tf.float32, name = "roi_mask_loss_{0}".format(stage + 1) if 1 < n_stage else "roi_mask_loss")(train_tag["mask_inputs"][stage], train_tag["mask_outputs"][stage])
            loss_roi_mask_list.append(loss_roi_mask * stage_weight[stage])
    
    loss_semantic = None
    if semantic_pred is not None:
        loss_semantic = FusedSemanticLoss(loss = semantic_loss, weight = class_weight, method = method,
                                          missing_value = missing_value, dtype = tf.float32, name = "semantic_loss")([y_true, mask_true], semantic_pred)
        loss_semantic = loss_semantic * semantic_weight
    
    input = [input] + [arg for arg in [y_true, bbox_true, mask_true] if arg is not None]
    args = [arg for arg in [cls_y_pred, cls_bbox_pred, proposals, cls_mask_pred] if arg is not None]
    out = FilterDetection(proposal_count = proposal_count, iou_threshold = iou_threshold, score_threshold = score_threshold, soft_nms = soft_nms, ensemble = ensemble, valid_inside_anchor = valid_inside_anchor, ignore_label = ignore_label, performance_count = performance_count,
                           mean = cls_mean, std = cls_std, clip_ratio = cls_clip_ratio,
                           batch_size = batch_size, dtype = tf.float32, name = "filter_detection")(args)
    model = tf.keras.Model(input, list(out))
    
    losses = []
    loss_rpn_class = [loss_rpn_class] if not isinstance(loss_rpn_class, (tuple, list)) else loss_rpn_class
    loss_rpn_bbox = [loss_rpn_bbox] if not isinstance(loss_rpn_bbox, (tuple, list)) else loss_rpn_bbox
    #for _loss_rpn_class, _loss_rpn_bbox in zip(loss_rpn_class, loss_rpn_bbox):
    #    model.add_loss(tf.expand_dims(_loss_rpn_class, axis = -1))
    #    model.add_loss(tf.expand_dims(_loss_rpn_bbox, axis = -1))
        
    loss_rpn_class = tf.expand_dims(tf.reduce_sum(loss_rpn_class), axis = -1)
    loss_rpn_bbox = tf.expand_dims(tf.reduce_sum(loss_rpn_bbox), axis = -1)
    model.add_metric(loss_rpn_class, name = "loss_rpn_class", aggregation = "mean")
    model.add_metric(loss_rpn_bbox, name = "loss_rpn_bbox", aggregation = "mean")
    losses += [loss_rpn_class, loss_rpn_bbox]
        
    for stage, (loss_roi_class, loss_roi_bbox) in enumerate(zip(loss_roi_class_list, loss_roi_bbox_list)):
        loss_roi_class = tf.expand_dims(loss_roi_class, axis = -1)
        loss_roi_bbox = tf.expand_dims(loss_roi_bbox, axis = -1)
        #model.add_loss(loss_roi_class)
        #model.add_loss(loss_roi_bbox)
        model.add_metric(loss_roi_class, name = "loss_roi_class_{0}".format(stage + 1) if 1 < n_stage else "loss_roi_class", aggregation = "mean")
        model.add_metric(loss_roi_bbox, name = "loss_roi_bbox_{0}".format(stage + 1) if 1 < n_stage else "loss_roi_bbox", aggregation = "mean")
        losses += [loss_roi_class, loss_roi_bbox]
        if 0 < len(loss_roi_mask_list):
            loss_roi_mask = tf.expand_dims(loss_roi_mask_list[stage], axis = -1)
            #model.add_loss(loss_roi_mask)
            model.add_metric(loss_roi_mask, name = "loss_roi_mask_{0}".format(stage + 1) if 1 < n_stage else "loss_roi_mask", aggregation = "mean")
            losses += [loss_roi_mask]
            
    if loss_semantic is not None:
        loss_semantic = tf.expand_dims(loss_semantic, axis = -1)
        #model.add_loss(loss_semantic)
        model.add_metric(loss_semantic, name = "loss_semantic", aggregation = "mean")
        losses += [loss_semantic]
        
    losses = tf.reduce_sum(losses, axis = 0)
    model.add_loss(losses)

    if regularize:
        model.add_loss(lambda: tf.cast(tf.reduce_sum(regularize_loss(model, weight_decay), keepdims = True), tf.float32))
    return model