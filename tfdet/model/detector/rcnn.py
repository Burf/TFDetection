import tensorflow as tf
import functools
import numpy as np

from tfdet.core.assign import max_iou
from tfdet.core.util import map_fn
from ..head import rpn_head, Rpn2Proposal, faster_rcnn_head, mask_rcnn_head, cascade_rcnn_head, hybrid_task_cascade_rcnn_head
from ..neck import FeatureAlign, fpn
from ..train.target import sampling_target

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def neck(n_feature = 256, n_sampling = 1, pre_sampling = False, neck = fpn, neck_n_depth = 1, convolution = conv, normalize = tf.keras.layers.BatchNormalization, **kwargs):
    return FeatureAlign(n_feature = n_feature, n_sampling = n_sampling, pre_sampling = pre_sampling, neck = neck, neck_n_depth = neck_n_depth, convolution = convolution, normalize = normalize, **kwargs)

def assign(bbox_true, bbox_pred, positive_threshold = 0.5, negative_threshold = 0.5, min_threshold = 0.5, match_low_quality = False, mode = "normal"):
    return max_iou(bbox_true, bbox_pred, positive_threshold = positive_threshold, negative_threshold = negative_threshold, min_threshold = min_threshold, match_low_quality = match_low_quality, mode = mode)

def rcnn(feature, neck = neck, rpn_head = rpn_head, cls_head = faster_rcnn_head,
         proposal_count = 1000, iou_threshold = 0.7, soft_nms = False, performance_count = 5000,
         mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, batch_size = 1,
         sampling_assign = assign, sampling_count = None, sampling_positive_ratio = 0.25, sampling_mask = False):
    """
    for speed training(with roi sampling) > sampling_count = 256 (Recommended value for training to apply prior to roi sampling)> return rcnn_layers(rpn, cls, ...) + sampling_tag(sampling_tag is a argument for rcnn_train_model)
    """
    if not isinstance(feature, list):
        feature = [feature]
    feature = list(feature)
    feature_count = len(feature)
    
    if neck is not None:
        feature = neck(name = "neck")(feature)
    
    rpn_score, rpn_regress, anchors = rpn_head(feature)
    proposals = Rpn2Proposal(proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, valid = False, performance_count = performance_count, batch_size = batch_size, mean = mean, std = std, clip_ratio = clip_ratio, name = "rpn2proposal")([rpn_score, rpn_regress], anchors)
    
    sampling_tag = None
    if isinstance(sampling_count, int) and 0 < sampling_count:
        y_true = tf.keras.layers.Input(shape = (None, None), name = "y_true", dtype = rpn_score.dtype) #(batch_size, padded_num_true, 1 or n_class)
        bbox_true = tf.keras.layers.Input(shape = (None, 4), name = "bbox_true", dtype = rpn_regress.dtype) #(batch_size, padded_num_true, 4)
        mask_true = tf.keras.layers.Input(shape = (None, None, None), name = "mask_true", dtype = rpn_score.dtype) if sampling_mask else None #(batch_size, padded_num_true, h, w)

        sampling_tag = {"sampling_assign":sampling_assign, "sampling_count":sampling_count, "positive_ratio":sampling_positive_ratio, "y_true":y_true, "bbox_true":bbox_true, "mask_true":mask_true}
        args = [y_true, bbox_true, proposals]
        dtype = (y_true.dtype, bbox_true.dtype, proposals.dtype)
        if sampling_mask:
            args = [y_true, bbox_true, proposals, mask_true]
            dtype = (y_true.dtype, bbox_true.dtype, mask_true.dtype, proposals.dtype)

        sampling_out = tf.keras.layers.Lambda(lambda args: map_fn(sampling_target, *args, dtype = dtype, batch_size = batch_size,
                                                                  assign = sampling_assign, sampling_count = sampling_count, positive_ratio = sampling_positive_ratio), name = "sampling_target")(args)
        sampling_out, proposals = sampling_out[:-1], sampling_out[-1]
        if len(sampling_out) == 2:
            sampling_out = [*sampling_out, None]
        sampling_y_true, sampling_bbox_true, sampling_mask_true = sampling_out
        sampling_tag.update({"sampling_y_true":sampling_y_true, "sampling_bbox_true":sampling_bbox_true, "sampling_mask_true":sampling_mask_true})
    
    feature = feature[:feature_count]
    cls_out = cls_head(feature, proposals)
    result = [rpn_score, rpn_regress, anchors, *cls_out]
    if sampling_tag is not None:
        result += [sampling_tag]
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
    cls_head = functools.partial(faster_rcnn_head, n_class = n_class, image_shape = image_shape, n_feature = cls_n_feature,
                                 pool_size = pool_size, method = method,
                                 convolution = cls_convolution, normalize = cls_normalize, activation = cls_activation)
    return rcnn(feature, neck = neck, rpn_head = _rpn_head, cls_head = cls_head,
                proposal_count = proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, performance_count = performance_count,
                mean = mean, std = std, clip_ratio = clip_ratio, batch_size = batch_size,
                sampling_assign = sampling_assign, sampling_count = sampling_count, sampling_positive_ratio = sampling_positive_ratio, sampling_mask = False)

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
    cls_head = functools.partial(mask_rcnn_head, n_class = n_class, image_shape = image_shape,
                                 cls_n_feature = cls_n_feature, mask_n_feature = mask_n_feature, mask_n_depth = mask_n_depth,
                                 pool_size = pool_size, method = method,
                                 cls_convolution = cls_convolution, cls_normalize = cls_normalize, cls_activation = cls_activation,
                                 mask_convolution = mask_convolution, mask_normalize = mask_normalize, mask_activation = mask_activation)
    return rcnn(feature, neck = neck, rpn_head = _rpn_head, cls_head = cls_head,
                proposal_count = proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, performance_count = performance_count,
                mean = mean, std = std, clip_ratio = clip_ratio, batch_size = batch_size,
                sampling_assign = sampling_assign, sampling_count = sampling_count, sampling_positive_ratio = sampling_positive_ratio, sampling_mask = True)

def cascade_rcnn(feature, n_class = 21, image_shape = [1024, 1024], mask = False,
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
    cls_head = functools.partial(cascade_rcnn_head, n_class = n_class, image_shape = image_shape, mask = mask,
                                 cls_n_feature = cls_n_feature, mask_n_feature = mask_n_feature, mask_n_depth = mask_n_depth,
                                 pool_size = pool_size, method = method,
                                 cls_convolution = cls_convolution, cls_normalize = cls_normalize, cls_activation = cls_activation,
                                 mask_convolution = mask_convolution, mask_normalize = mask_normalize, mask_activation = mask_activation)
    return rcnn(feature, neck = neck, rpn_head = _rpn_head, cls_head = cls_head,
                proposal_count = proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, performance_count = performance_count,
                mean = mean, std = std, clip_ratio = clip_ratio, batch_size = batch_size,
                sampling_assign = sampling_assign, sampling_count = sampling_count, sampling_positive_ratio = sampling_positive_ratio, sampling_mask = mask)

def hybrid_task_cascade_rcnn(feature, n_class = 21, image_shape = [1024, 1024], mask = True, mask_info_flow = True, semantic_feature = True,
                             scale = [0.03125, 0.0625, 0.125, 0.25, 0.5], ratio = [0.5, 1, 2], auto_scale = True,
                             rpn_n_feature = 256, rpn_use_bias = True, rpn_feature_share = True,
                             cls_n_feature = 1024, mask_n_feature = 256, mask_n_depth = 4, semantic_level = 1, semantic_n_feature = 256, semantic_n_depth = 4, 
                             proposal_count = 1000, iou_threshold = 0.7, soft_nms = False, performance_count = 5000,
                             pool_size = 7, semantic_pool_size = 14, method = "bilinear",
                             mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, batch_size = 1,
                             sampling_assign = assign, sampling_count = None, sampling_positive_ratio = 0.25,
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
    cls_head = functools.partial(hybrid_task_cascade_rcnn_head, n_class = n_class, image_shape = image_shape, mask = mask, mask_info_flow = mask_info_flow, semantic_feature = semantic_feature,
                                 cls_n_feature = cls_n_feature, mask_n_feature = mask_n_feature, mask_n_depth = mask_n_depth, semantic_level = semantic_level, semantic_n_feature = semantic_n_feature, semantic_n_depth = semantic_n_depth,
                        pool_size = pool_size, semantic_pool_size = semantic_pool_size, method = method,
                                 cls_convolution = cls_convolution, cls_normalize = cls_normalize, cls_activation = cls_activation,
                                 mask_convolution = mask_convolution, mask_normalize = mask_normalize, mask_activation = mask_activation,
                                 semantic_logits_activation = semantic_logits_activation, semantic_convolution = semantic_convolution, semantic_normalize = semantic_normalize, semantic_activation = semantic_activation)
    return rcnn(feature, neck = neck, rpn_head = _rpn_head, cls_head = cls_head,
                proposal_count = proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, performance_count = performance_count,
                mean = mean, std = std, clip_ratio = clip_ratio, batch_size = batch_size,
                sampling_assign = sampling_assign, sampling_count = sampling_count, sampling_positive_ratio = sampling_positive_ratio, sampling_mask = mask)