import tensorflow as tf
import functools
import numpy as np

from tfdet.core.assign import max_iou
from tfdet.core.util import map_fn
from ..head import rpn_head, bbox_head, mask_head, semantic_head, Rpn2Proposal, Classifier2Proposal
from ..neck import FeatureAlign, fpn
from ..train.target import sampling_target

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def neck(n_feature = 256, n_sampling = 1, pre_sampling = False, neck = fpn, neck_n_depth = 1, convolution = conv, normalize = tf.keras.layers.BatchNormalization, **kwargs):
    return FeatureAlign(n_feature = n_feature, n_sampling = n_sampling, pre_sampling = pre_sampling, neck = neck, neck_n_depth = neck_n_depth, convolution = convolution, normalize = normalize, **kwargs)

def assign(bbox_true, bbox_pred, positive_threshold = 0.5, negative_threshold = 0.5, min_threshold = 0.5, match_low_quality = False, mode = "normal"):
    return max_iou(bbox_true, bbox_pred, positive_threshold = positive_threshold, negative_threshold = negative_threshold, min_threshold = min_threshold, match_low_quality = match_low_quality, mode = mode)

def assign2(bbox_true, bbox_pred, positive_threshold = 0.6, negative_threshold = 0.6, min_threshold = 0.6, match_low_quality = False, mode = "normal"):
    return max_iou(bbox_true, bbox_pred, positive_threshold = positive_threshold, negative_threshold = negative_threshold, min_threshold = min_threshold, match_low_quality = match_low_quality, mode = mode)

def assign3(bbox_true, bbox_pred, positive_threshold = 0.7, negative_threshold = 0.7, min_threshold = 0.7, match_low_quality = False, mode = "normal"):
    return max_iou(bbox_true, bbox_pred, positive_threshold = positive_threshold, negative_threshold = negative_threshold, min_threshold = min_threshold, match_low_quality = match_low_quality, mode = mode)

def rcnn(feature, neck = neck, rpn_head = rpn_head, bbox_head = bbox_head, mask_head = None, semantic_head = None,
         cascade = False, mask_info_flow = False,
         proposal_count = 1000, iou_threshold = 0.7, soft_nms = False, performance_count = 5000,
         mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, batch_size = 1,
         sampling_assign = [assign, assign2, assign3], sampling_count = None, sampling_positive_ratio = 0.25):
    """
    for speed training(with roi sampling) > sampling_count = 256 (Recommended value for training to apply prior to roi sampling)> return rcnn_layers(rpn, cls, ...) + sampling_tag(sampling_tag is a argument for rcnn_train_model)
    """
    if not isinstance(feature, list):
        feature = [feature]
    if not isinstance(sampling_assign, list):
        sampling_assign = [sampling_assign]
    feature = list(feature)
    feature_count = len(feature)
    sampling_assign = sampling_assign * 3 if cascade and len(sampling_assign) == 1 else sampling_assign
    
    if neck is not None:
        feature = neck(name = "neck")(feature)
    
    rpn_score, rpn_regress, anchors = rpn_head(feature)
    proposals = Rpn2Proposal(proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, valid = False, performance_count = performance_count, batch_size = batch_size, mean = mean, std = std, clip_ratio = clip_ratio, name = "proposals")([rpn_score, rpn_regress], anchors)
    
    sampling_tag = None
    if isinstance(sampling_count, int) and 0 < sampling_count:
        sampling_y_true = y_true = tf.keras.layers.Input(shape = (None, None), name = "y_true", dtype = rpn_score.dtype) #(batch_size, padded_num_true, 1 or n_class)
        sampling_bbox_true = bbox_true = tf.keras.layers.Input(shape = (None, 4), name = "bbox_true", dtype = rpn_regress.dtype) #(batch_size, padded_num_true, 4)
        sampling_mask_true = mask_true = tf.keras.layers.Input(shape = (None, None, None), name = "mask_true", dtype = rpn_score.dtype) if mask_head is not None or semantic_head is not None else None #(batch_size, padded_num_true, h, w)
        
        sampling_func = lambda args, **kwarg: map_fn(sampling_target, *args, dtype = (sampling_y_true.dtype, sampling_bbox_true.dtype, sampling_bbox_true.dtype), batch_size = batch_size, **kwargs)
        sampling_mask_func = lambda args, **kwarg: map_fn(sampling_target, *args, dtype = (sampling_y_true.dtype, sampling_bbox_true.dtype, sampling_mask_true.dtype, sampling_bbox_true.dtype), batch_size = batch_size, **kwargs)
        
        sampling_tag = {"sampling_assign":sampling_assign, "sampling_count":sampling_count, "positive_ratio":sampling_positive_ratio, "y_true":y_true, "bbox_true":bbox_true, "mask_true":mask_true, "sampling_y_true":[], "sampling_bbox_true":[], "sampling_mask_true":[]}
    
    n_stage = (3 if cascade else 1) + (1 if mask_head is not None and mask_info_flow else 0)
    feature = feature[:feature_count]
    mask_feature = semantic_regress = semantic_feature = None
    if semantic_head is not None:
        semantic_regress, semantic_feature = semantic_head(feature)
    cls_logits, cls_regress, proposals, mask_regress = [], [], [proposals], []
    for stage in range(n_stage):
        _proposals = proposals[-1]
        if sampling_tag is not None and not (mask_head is not None and mask_info_flow and (stage + 1) == n_stage):
            kwargs = {"assign":sampling_assign[stage], "sampling_count":sampling_count, "positive_ratio":sampling_positive_ratio}
            if mask_head is not None and not (mask_info_flow and stage == 0):
                sampling_y_true, sampling_bbox_true, sampling_mask_true, _proposals = tf.keras.layers.Lambda(sampling_mask_func, arguments = kwargs, name = "sampling_target_{0}".format(stage + 1))([sampling_y_true, sampling_bbox_true, _proposals, sampling_mask_true])
                sampling_tag["sampling_mask_true"].append(sampling_mask_true)
            else:
                sampling_y_true, sampling_bbox_true, _proposals = tf.keras.layers.Lambda(sampling_func, arguments = kwargs, name = "sampling_target_{0}".format(stage + 1))([sampling_y_true, sampling_bbox_true, _proposals])
                sampling_tag["sampling_mask_true"].append(None)
            sampling_tag["sampling_y_true"].append(sampling_y_true)
            sampling_tag["sampling_bbox_true"].append(sampling_bbox_true)
            proposals[-1] = _proposals
        
        if not (mask_head is not None and mask_info_flow and (stage + 1) == n_stage):
            _cls_logits, _cls_regress = bbox_head(feature, _proposals, semantic_feature = semantic_feature)
            cls_logits.append(_cls_logits)
            cls_regress.append(_cls_regress)
        
        if mask_head is not None:
            if mask_info_flow:
                if 0 < stage:
                    _mask_regress, mask_feature = mask_head(feature, _proposals, mask_feature = mask_feature, semantic_feature = semantic_feature)
                    mask_regress.append(_mask_regress)
            else:
                _mask_regress, _ = mask_head(feature, _proposals, semantic_feature = semantic_feature)
                mask_regress.append(_mask_regress)
        
        if (stage + 1) < n_stage:
            _proposals = Classifier2Proposal(True, batch_size, mean, std, clip_ratio, name = "proposals_{0}".format(stage + 2))([cls_logits[-1], cls_regress[-1], proposals[-1]])
            proposals.append(_proposals)
    
    if len(mask_regress) == 0:
        mask_regress = None
    cls_logits, cls_regress, proposals, mask_regress = [r[0] if r is not None and len(r) == 1 else r for r in [cls_logits, cls_regress, proposals, mask_regress]]
    result = [r for r in [rpn_score, rpn_regress, anchors, cls_logits, cls_regress, proposals, mask_regress, semantic_regress, sampling_tag] if r is not None]
    return result

def faster_rcnn(feature, n_class = 21, image_shape = [1024, 1024], 
                scale = [0.03125, 0.0625, 0.125, 0.25, 0.5], ratio = [0.5, 1, 2], auto_scale = True,
                rpn_n_feature = 256, rpn_use_bias = True, rpn_feature_share = True,
                cls_n_feature = 1024,
                proposal_count = 1000, iou_threshold = 0.7, soft_nms = False, performance_count = 5000,
                pool_size = 7, method = "bilinear",
                mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, batch_size = 1,
                sampling_assign = assign, sampling_count = None, sampling_positive_ratio = 0.25,
                neck = neck,
                rpn_convolution = conv, rpn_normalize = None, rpn_activation = tf.keras.activations.relu,
                cls_convolution = conv, cls_normalize = tf.keras.layers.BatchNormalization, cls_activation = tf.keras.activations.relu):
    """
    for speed training(with roi sampling) > sampling_count = 256 (Recommended value for training to apply prior to roi sampling)> return rcnn_layers(rpn, cls, ...) + sampling_tag(sampling_tag is a argument for rcnn_train_model)
    """
    _rpn_head = functools.partial(rpn_head, image_shape = image_shape,
                                  scale = scale, ratio = ratio, auto_scale = auto_scale,
                                  n_feature = rpn_n_feature, use_bias = rpn_use_bias, feature_share = rpn_feature_share,
                                  convolution = rpn_convolution, normalize = rpn_normalize, activation = rpn_activation)
    _bbox_head = functools.partial(bbox_head, n_class = n_class, image_shape = image_shape, n_feature = cls_n_feature,
                                   pool_size = pool_size, method = method,
                                   convolution = cls_convolution, normalize = cls_normalize, activation = cls_activation)
    return rcnn(feature, neck = neck, rpn_head = _rpn_head, bbox_head = _bbox_head,
                proposal_count = proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, performance_count = performance_count,
                mean = mean, std = std, clip_ratio = clip_ratio, batch_size = batch_size,
                sampling_assign = sampling_assign, sampling_count = sampling_count, sampling_positive_ratio = sampling_positive_ratio)

def mask_rcnn(feature, n_class = 21, image_shape = [1024, 1024], 
              scale = [0.03125, 0.0625, 0.125, 0.25, 0.5], ratio = [0.5, 1, 2], auto_scale = True,
              rpn_n_feature = 256, rpn_use_bias = True, rpn_feature_share = True,
              cls_n_feature = 1024, mask_n_feature = 256, mask_n_depth = 4, 
              proposal_count = 1000, iou_threshold = 0.7, soft_nms = False, performance_count = 5000,
              pool_size = 7, method = "bilinear",
              mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, batch_size = 1,
              sampling_assign = assign, sampling_count = None, sampling_positive_ratio = 0.25,
              neck = neck,
              rpn_convolution = conv, rpn_normalize = None, rpn_activation = tf.keras.activations.relu,
              cls_convolution = conv, cls_normalize = tf.keras.layers.BatchNormalization, cls_activation = tf.keras.activations.relu,
              mask_convolution = conv, mask_normalize = tf.keras.layers.BatchNormalization, mask_activation = tf.keras.activations.relu):
    """
    for speed training(with roi sampling) > sampling_count = 256 (Recommended value for training to apply prior to roi sampling)> return rcnn_layers(rpn, cls, ...) + sampling_tag(sampling_tag is a argument for rcnn_train_model)
    """
    _rpn_head = functools.partial(rpn_head, image_shape = image_shape,
                                  scale = scale, ratio = ratio, auto_scale = auto_scale,
                                  n_feature = rpn_n_feature, use_bias = rpn_use_bias, feature_share = rpn_feature_share,
                                  convolution = rpn_convolution, normalize = rpn_normalize, activation = rpn_activation)
    _bbox_head = functools.partial(bbox_head, n_class = n_class, image_shape = image_shape, n_feature = cls_n_feature, 
                                   pool_size = pool_size, method = method,
                                   convolution = cls_convolution, normalize = cls_normalize, activation = cls_activation)
    _mask_head = functools.partial(mask_head, n_class = n_class, image_shape = image_shape, n_feature = mask_n_feature, n_depth = mask_n_depth,
                                   pool_size = pool_size, method = method,
                                   convolution = mask_convolution, normalize = mask_normalize, activation = mask_activation)
    return rcnn(feature, neck = neck, rpn_head = _rpn_head, bbox_head = _bbox_head, mask_head = _mask_head,
                proposal_count = proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, performance_count = performance_count,
                mean = mean, std = std, clip_ratio = clip_ratio, batch_size = batch_size,
                sampling_assign = sampling_assign, sampling_count = sampling_count, sampling_positive_ratio = sampling_positive_ratio)

def cascade_rcnn(feature, n_class = 21, image_shape = [1024, 1024], mask = False,
                 scale = [0.03125, 0.0625, 0.125, 0.25, 0.5], ratio = [0.5, 1, 2], auto_scale = True,
                 rpn_n_feature = 256, rpn_use_bias = True, rpn_feature_share = True,
                 cls_n_feature = 1024, mask_n_feature = 256, mask_n_depth = 4, 
                 proposal_count = 1000, iou_threshold = 0.7, soft_nms = False, performance_count = 5000,
                 pool_size = 7, method = "bilinear",
                 mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, batch_size = 1,
                 sampling_assign = [assign, assign2, assign3], sampling_count = None, sampling_positive_ratio = 0.25,
                 neck = neck,
                 rpn_convolution = conv, rpn_normalize = None, rpn_activation = tf.keras.activations.relu,
                 cls_convolution = conv, cls_normalize = tf.keras.layers.BatchNormalization, cls_activation = tf.keras.activations.relu,
                 mask_convolution = conv, mask_normalize = tf.keras.layers.BatchNormalization, mask_activation = tf.keras.activations.relu):
    """
    for speed training(with roi sampling) > sampling_count = 256 (Recommended value for training to apply prior to roi sampling)> return rcnn_layers(rpn, cls, ...) + sampling_tag(sampling_tag is a argument for rcnn_train_model)
    """
    _rpn_head = functools.partial(rpn_head, image_shape = image_shape,
                                  scale = scale, ratio = ratio, auto_scale = auto_scale,
                                  n_feature = rpn_n_feature, use_bias = rpn_use_bias, feature_share = rpn_feature_share,
                                  convolution = rpn_convolution, normalize = rpn_normalize, activation = rpn_activation)
    _bbox_head = functools.partial(bbox_head, n_class = n_class, image_shape = image_shape, n_feature = cls_n_feature, 
                                   pool_size = pool_size, method = method,
                                   convolution = cls_convolution, normalize = cls_normalize, activation = cls_activation)
    _mask_head = None
    if mask:
        _mask_head = functools.partial(mask_head, n_class = n_class, image_shape = image_shape, n_feature = mask_n_feature, n_depth = mask_n_depth,
                                       pool_size = pool_size, method = method,
                                       convolution = mask_convolution, normalize = mask_normalize, activation = mask_activation)
    return rcnn(feature, neck = neck, rpn_head = _rpn_head, bbox_head = _bbox_head, mask_head = _mask_head,
                cascade = True,
                proposal_count = proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, performance_count = performance_count,
                mean = mean, std = std, clip_ratio = clip_ratio, batch_size = batch_size,
                sampling_assign = sampling_assign, sampling_count = sampling_count, sampling_positive_ratio = sampling_positive_ratio)
    
def hybrid_task_cascade_rcnn(feature, n_class = 21, image_shape = [1024, 1024], mask = True, mask_info_flow = True, semantic_feature = True,
                             scale = [0.03125, 0.0625, 0.125, 0.25, 0.5], ratio = [0.5, 1, 2], auto_scale = True,
                             rpn_n_feature = 256, rpn_use_bias = True, rpn_feature_share = True,
                             cls_n_feature = 1024, mask_n_feature = 256, mask_n_depth = 4, semantic_level = 1, semantic_n_feature = 256, semantic_n_depth = 4, 
                             proposal_count = 1000, iou_threshold = 0.7, soft_nms = False, performance_count = 5000,
                             pool_size = 7, semantic_pool_size = 14, method = "bilinear",
                             mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, batch_size = 1,
                             sampling_assign = [assign, assign2, assign3], sampling_count = None, sampling_positive_ratio = 0.25,
                             neck = neck,
                             rpn_convolution = conv, rpn_normalize = None, rpn_activation = tf.keras.activations.relu,
                             cls_convolution = conv, cls_normalize = tf.keras.layers.BatchNormalization, cls_activation = tf.keras.activations.relu,
                             mask_convolution = conv, mask_normalize = tf.keras.layers.BatchNormalization, mask_activation = tf.keras.activations.relu,
                             semantic_logits_activation = None, semantic_convolution = conv, semantic_normalize = None, semantic_activation = tf.keras.activations.relu):
    """
    for speed training(with roi sampling) > sampling_count = 256 (Recommended value for training to apply prior to roi sampling)> return rcnn_layers(rpn, cls, ...) + sampling_tag(sampling_tag is a argument for rcnn_train_model)
    """
    _rpn_head = functools.partial(rpn_head, image_shape = image_shape,
                                  scale = scale, ratio = ratio, auto_scale = auto_scale,
                                  n_feature = rpn_n_feature, use_bias = rpn_use_bias, feature_share = rpn_feature_share,
                                  convolution = rpn_convolution, normalize = rpn_normalize, activation = rpn_activation)
    _bbox_head = functools.partial(bbox_head, n_class = n_class, image_shape = image_shape, n_feature = cls_n_feature, 
                                   pool_size = pool_size, semantic_pool_size = semantic_pool_size, method = method,
                                   convolution = cls_convolution, normalize = cls_normalize, activation = cls_activation)
    _semantic_head = _mask_head = None
    if mask:
        _mask_head = functools.partial(mask_head, n_class = n_class, image_shape = image_shape, n_feature = mask_n_feature, n_depth = mask_n_depth,
                                       pool_size = pool_size, semantic_pool_size = semantic_pool_size, method = method,
                                       convolution = mask_convolution, normalize = mask_normalize, activation = mask_activation)
    if semantic_feature:
        _semantic_head = functools.partial(semantic_head, n_class = n_class, level = semantic_level, n_feature = semantic_n_feature, n_depth = semantic_n_depth, method = method, logits_activation = semantic_logits_activation, convolution = semantic_convolution, normalize = semantic_normalize, activation = semantic_activation)
    return rcnn(feature, neck = neck, rpn_head = _rpn_head, bbox_head = _bbox_head, mask_head = _mask_head, semantic_head = _semantic_head,
                cascade = True, mask_info_flow = mask_info_flow,
                proposal_count = proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, performance_count = performance_count,
                mean = mean, std = std, clip_ratio = clip_ratio, batch_size = batch_size,
                sampling_assign = sampling_assign, sampling_count = sampling_count, sampling_positive_ratio = sampling_positive_ratio)
