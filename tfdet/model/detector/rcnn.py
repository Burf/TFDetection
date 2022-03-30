import tensorflow as tf
import numpy as np

from tfdet.core.target import sampling_target
from tfdet.core.util.anchor import generate_anchors
from tfdet.core.util.tf import map_fn
from ..head.rcnn import RegionProposalNetwork, Rpn2Proposal, RoiAlign, RoiClassifier, RoiMask, Classifier2Proposal, FusedSemanticHead
from ..neck import fpn

def rcnn(feature, n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4, sub_sampling = 1,
         scale = [0.03125, 0.0625, 0.125, 0.25, 0.5], ratio = [0.5, 1, 2], auto_scale = True,
         mask = False, cascade = True, mask_info_flow = False, semantic_feature = False,
         proposal_count = 1000, iou_threshold = 0.7, soft_nms = True, valid = True, performance_count = 5000,
         pool_size = 7, method = "bilinear",
         mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, batch_size = 1,
         sub_n_feature = None, sub_momentum = 0.997, sub_epsilon = 1e-4, fpn = fpn, fpn_n_depth = 1,
         rpn_feature_share = True, rpn_n_feature = None, rpn_use_bias = False, rpn_activation = tf.keras.activations.relu,
         cls_n_feature = None, cls_activation = tf.keras.activations.relu,
         mask_n_feature = None, mask_n_depth = None, mask_activation = tf.keras.activations.relu,
         semantic_level = 1, semantic_n_feature = None, semantic_n_depth = None, semantic_pool_size = 14, semantic_activation = tf.keras.activations.relu,
         sampling_count = None, sampling_positive_ratio = 0.25, sampling_positive_threshold = 0.5, sampling_negative_threshold = 0.5,
         **kwargs):
    """
    with single feature                   > feature = single feature
    with fpn                              > feature = multi feature, sub_sampling = len(scale) - len(feature)
    for speed training(with roi sampling) > sampling_tag = 256 (recommendation value for train sampling)                    > return rcnn_layers(rpn, cls, ...) + sampling_tag(sampling_tag is a argument for rcnn_train_model)
    faster rcnn                           > mask = False, cascade = False, mask_info_flow = False, semantic_feature = False > return rpn_score, rpn_regress, cls_logits, cls_regress, proposals, anchors
    mask rcnn                             > mask = True, cascade = False, mask_info_flow = False, semantic_feature = False  > return rpn_score, rpn_regress, cls_logits, cls_regress, proposals, anchors, mask_regress
    cascade rcnn                          > cascade = True, mask_info_flow = False, semantic_feature = False                > return rpn_score, rpn_regress, cls_logits, cls_regress, proposals, anchors, mask_regress(mask = True)
    hybrid task cascade rcnn              > cascade = True, mask_info_flow = True, semantic_feature = True                  > return rpn_score, rpn_regress, cls_logits, cls_regress, proposals, anchors, mask_regress(mask = True), semantic_regress
    """
    if tf.is_tensor(image_shape) and 2 < tf.keras.backend.ndim(image_shape) or (not tf.is_tensor(image_shape) and 2 < np.ndim(image_shape)):
        image_shape = tf.shape(image_shape) if tf.keras.backend.int_shape(image_shape)[-3] is None else tf.keras.backend.int_shape(image_shape)
    if 2 < np.shape(image_shape)[0]:
        image_shape = image_shape[-3:-1]
    if not isinstance(feature, list):
        feature = [feature]
    feature = list(feature)
    sub_n_feature = sub_n_feature if sub_n_feature is not None else n_feature
    rpn_n_feature = rpn_n_feature if rpn_n_feature is not None else n_feature * 2
    cls_n_feature = cls_n_feature if cls_n_feature is not None else n_feature * 4
    mask_n_feature = mask_n_feature if mask_n_feature is not None else n_feature
    semantic_n_feature = semantic_n_feature if semantic_n_feature is not None else n_feature
    mask_n_depth = mask_n_depth if mask_n_depth is not None else n_depth
    semantic_n_depth = semantic_n_depth if semantic_n_depth is not None else n_depth
    
    if fpn_n_depth < 1:
        feature = [tf.keras.layers.Conv2D(n_feature, 1, use_bias = True, kernel_initializer = "he_normal", name = "feature_resample_conv{0}".format(i + 1) if 1 < len(feature) else "feature_resample_conv")(x) for i, x in enumerate(feature)]
    else:
        for index in range(fpn_n_depth):
            feature = fpn(name = "feature_pyramid_network{0}".format(index + 1) if 1 < fpn_n_depth else "feature_pyramid_network")(feature)
    for index in range(sub_sampling):
        #feature.append(tf.keras.layers.MaxPooling2D((1, 1), strides = 2, name = "feature_sub_sampling{0}".format(index + 1) if 1 < sub_sampling else "feature_sub_sampling")(feature[-1]))
        x = feature[-1]
        if index == 0:
            x = tf.keras.layers.Conv2D(sub_n_feature, 1, use_bias = False, name = "feature_sub_sampling_pre_conv")(x)
            x = tf.keras.layers.BatchNormalization(axis = -1, momentum = sub_momentum, epsilon = sub_epsilon, name = "feature_sub_sampling_pre_norm")(x)
        feature.append(tf.keras.layers.MaxPooling2D((3, 3), strides = 2, padding = "same", name = "feature_sub_sampling{0}".format(index + 1) if 1 < sub_sampling else "feature_sub_sampling")(x))
        
    n_anchor = len(scale) * len(ratio)
    if isinstance(scale, list) and isinstance(scale[0], list):
        n_anchor = len(scale[0]) * len(ratio)
    elif auto_scale and (len(scale) % len(feature)) == 0:
        n_anchor = (len(scale) // len(feature)) * len(ratio)
    rpn_score, rpn_regress = RegionProposalNetwork(n_anchor, rpn_feature_share, rpn_n_feature, rpn_use_bias, rpn_activation, name = "region_proposal_network")(feature)
    anchors = generate_anchors(feature, image_shape, scale, ratio, normalize = True, auto_scale = auto_scale)

    _proposals = Rpn2Proposal(proposal_count, iou_threshold, soft_nms, valid, performance_count, batch_size, mean, std, clip_ratio, name = "rpn_to_proposal")([rpn_score, rpn_regress], anchors)
    sampling_tag = None
    if isinstance(sampling_count, int) and 0 < sampling_count:
        y_true = tf.keras.layers.Input(shape = (None, None), name = "y_true", dtype = rpn_score.dtype) #(batch_size, padded_num_true, 1 or n_class)
        bbox_true = tf.keras.layers.Input(shape = (None, 4), name = "bbox_true", dtype = rpn_regress.dtype) #(batch_size, padded_num_true, 4)
        mask_true = tf.keras.layers.Input(shape = (None, None, None), name = "mask_true", dtype = rpn_score.dtype) if mask or semantic_feature else None #(batch_size, padded_num_true, h, w)
        sampling_tag = {"sampling_count":sampling_count, "positive_ratio":sampling_positive_ratio, "positive_threshold":sampling_positive_threshold, "negative_threshold":sampling_negative_threshold, "y_true":y_true, "bbox_true":bbox_true, "mask_true":mask_true}
        if mask:
            sampling_y_true, sampling_bbox_true, sampling_mask_true, _proposals = tf.keras.layers.Lambda(lambda args: map_fn(sampling_target, *args, dtype = (y_true.dtype, bbox_true.dtype, mask_true.dtype, _proposals.dtype), batch_size = batch_size,
                                                                                                                             sampling_count = sampling_count, positive_ratio = sampling_positive_ratio, positive_threshold = sampling_positive_threshold, negative_threshold = sampling_negative_threshold), name = "sampling_target")([y_true, bbox_true, _proposals, mask_true])
        else:
            sampling_y_true, sampling_bbox_true, _proposals = tf.keras.layers.Lambda(lambda args: map_fn(sampling_target, *args, dtype = (y_true.dtype, bbox_true.dtype, _proposals.dtype), batch_size = batch_size,
                                                                                                         sampling_count = sampling_count, positive_ratio = sampling_positive_ratio, positive_threshold = sampling_positive_threshold, negative_threshold = sampling_negative_threshold), name = "sampling_target")([y_true, bbox_true, _proposals])
            sampling_mask_true = None
        sampling_tag.update({"sampling_y_true":sampling_y_true, "sampling_bbox_true":sampling_bbox_true, "sampling_mask_true":sampling_mask_true})

    semantic_regress = None
    if semantic_feature:
        semantic_regress, _semantic_feature = FusedSemanticHead(n_class, semantic_n_feature, semantic_n_depth, activation = semantic_activation, name = "semantic_feature")(feature, min(semantic_level, len(feature) - 1), feature = True)
        semantic_roi_extractor = RoiAlign(semantic_pool_size, method, name = "semantic_roi_align")
        if pool_size != semantic_pool_size:
            semantic_resize = tf.keras.layers.TimeDistributed(tf.keras.layers.Lambda(lambda args: tf.image.resize(args, [pool_size, pool_size], method = "bilinear")), name = "semantic_resize_roi_align")
    
    n_stage = 3 if cascade else 1
    if mask and mask_info_flow:
        n_stage += 1
    feature = feature[:-sub_sampling] if 1 < len(feature) and 0 < sub_sampling else feature
    roi_extractor = RoiAlign(pool_size, method, name = "roi_align")
    mask_feature = None
    cls_logits, cls_regress, proposals, mask_regress = [], [], [_proposals], []
    for index in range(n_stage):
        roi = roi_extractor([feature, proposals[-1]], image_shape)
        if semantic_feature:
            semantic_roi = semantic_roi_extractor([[_semantic_feature], proposals[-1]], image_shape)
            if pool_size != semantic_pool_size:
                semantic_roi = semantic_resize(semantic_roi)
            roi = tf.keras.layers.Add(name = "roi_align_with_semantic_feature{0}".format(index + 1) if n_stage != 1 else "roi_align_with_semantic_feature")([roi, semantic_roi])
        
        if n_stage in [1, 3] or (n_stage in [2, 4] and index < (n_stage - 1)):
            _cls_logits, _cls_regress = RoiClassifier(n_class, cls_n_feature, cls_activation, name = "roi_classifier{0}".format(index + 1) if 2 < n_stage else "roi_classifier")(roi)
            cls_logits.append(_cls_logits)
            cls_regress.append(_cls_regress)
        
        if index < (n_stage - 1):
            proposals.append(Classifier2Proposal(True, batch_size, mean, std, clip_ratio, name = "classifier_to_proposal{0}".format(index + 1) if n_stage != 1 else "classifier_to_proposal")([_cls_logits, _cls_regress, proposals[-1]]))

        if mask:
            if mask_info_flow:
                if 0 < index:
                    _mask_regress, mask_feature = RoiMask(n_class, mask_n_feature, mask_n_depth, mask_activation, name = "roi_mask{0}".format(index) if 2 < n_stage else "roi_mask")([roi, mask_feature], feature = True)
                    mask_regress.append(_mask_regress)
            else:
                _mask_regress = RoiMask(n_class, mask_n_feature, mask_n_depth, mask_activation, name = "roi_mask{0}".format(index + 1) if 2 < n_stage else "roi_mask")(roi)
                mask_regress.append(_mask_regress)

    if len(mask_regress) == 0:
        mask_regress = None
    if n_stage < 3:
        cls_logits, cls_regress, mask_regress = cls_logits[0], cls_regress[0], mask_regress[0] if mask_regress is not None else None
        if len(proposals) == 1:
            proposals = proposals[0]
    
    valid_flags = tf.logical_and(tf.less_equal(anchors[..., 2], 1),
                                  tf.logical_and(tf.less_equal(anchors[..., 3], 1),
                                                tf.logical_and(tf.greater_equal(anchors[..., 0], 0),
                                                                tf.greater_equal(anchors[..., 1], 0))))
    #valid_indices = tf.range(tf.shape(rpn_score)[1])[valid_flags]
    valid_indices = tf.where(valid_flags)[:, 0]
    rpn_score = tf.gather(rpn_score, valid_indices, axis = 1)
    rpn_regress = tf.gather(rpn_regress, valid_indices, axis = 1)
    anchors = tf.gather(anchors, valid_indices)
    
    result = [r for r in [rpn_score, rpn_regress, cls_logits, cls_regress, proposals, anchors, mask_regress, semantic_regress, sampling_tag] if r is not None]
    return result

def faster_rcnn(feature, n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4, sub_sampling = 1,
                scale = [0.03125, 0.0625, 0.125, 0.25, 0.5], ratio = [0.5, 1, 2], auto_scale = True,
                mask = False, cascade = False, mask_info_flow = False, semantic_feature = False,
                proposal_count = 1000, iou_threshold = 0.7, soft_nms = True, valid = True, performance_count = 5000,
                pool_size = 7, method = "bilinear",
                mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, batch_size = 1,
                sub_n_feature = None, sub_momentum = 0.997, sub_epsilon = 1e-4, fpn = fpn, fpn_n_depth = 1,
                rpn_feature_share = True, rpn_n_feature = None, rpn_use_bias = False, rpn_activation = tf.keras.activations.relu,
                cls_n_feature = None, cls_activation = tf.keras.activations.relu,
                mask_n_feature = None, mask_n_depth = 4, mask_activation = tf.keras.activations.relu,
                semantic_level = 1, semantic_n_feature = None, semantic_n_depth = 4, semantic_pool_size = 14, semantic_activation = tf.keras.activations.relu,
                sampling_count = None, sampling_positive_ratio = 0.25, sampling_positive_threshold = 0.5, sampling_negative_threshold = 0.5):
    out = rcnn(feature, n_class = n_class, image_shape = image_shape, n_feature = n_feature, n_depth = n_depth, sub_sampling = sub_sampling,
               scale = scale, ratio = ratio, auto_scale = auto_scale,
               mask = mask, cascade = cascade, mask_info_flow = mask_info_flow, semantic_feature = semantic_feature,
               proposal_count = proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, valid = valid, performance_count = performance_count,
               pool_size = pool_size, method = method,
               mean = mean, std = std, clip_ratio = clip_ratio, batch_size = batch_size,
               sub_n_feature = sub_n_feature, sub_momentum = sub_momentum, sub_epsilon = sub_epsilon, fpn = fpn, fpn_n_depth = fpn_n_depth,
               rpn_feature_share = rpn_feature_share, rpn_n_feature = rpn_n_feature, rpn_use_bias = rpn_use_bias, rpn_activation = rpn_activation,
               cls_n_feature = cls_n_feature, cls_activation = cls_activation,
               mask_n_feature = mask_n_feature, mask_n_depth = mask_n_depth, mask_activation = mask_activation,
               semantic_level = semantic_level, semantic_n_feature = semantic_n_feature, semantic_n_depth = semantic_n_depth, semantic_pool_size = semantic_pool_size, semantic_activation = semantic_activation,
               sampling_count = sampling_count, sampling_positive_ratio = sampling_positive_ratio, sampling_positive_threshold = sampling_positive_threshold, sampling_negative_threshold = sampling_negative_threshold)
    return out

def mask_rcnn(feature, n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4, sub_sampling = 1,
              scale = [0.03125, 0.0625, 0.125, 0.25, 0.5], ratio = [0.5, 1, 2], auto_scale = True,
              mask = True, cascade = False, mask_info_flow = False, semantic_feature = False,
              proposal_count = 1000, iou_threshold = 0.7, soft_nms = True, valid = True, performance_count = 5000,
              pool_size = 7, method = "bilinear",
              mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, batch_size = 1,
              sub_n_feature = None, sub_momentum = 0.997, sub_epsilon = 1e-4, fpn = fpn, fpn_n_depth = 1,
              rpn_feature_share = True, rpn_n_feature = None, rpn_use_bias = False, rpn_activation = tf.keras.activations.relu,
              cls_n_feature = None, cls_activation = tf.keras.activations.relu,
              mask_n_feature = None, mask_n_depth = 4, mask_activation = tf.keras.activations.relu,
              semantic_level = 1, semantic_n_feature = None, semantic_n_depth = 4, semantic_pool_size = 14, semantic_activation = tf.keras.activations.relu,
              sampling_count = None, sampling_positive_ratio = 0.25, sampling_positive_threshold = 0.5, sampling_negative_threshold = 0.5):
    out = rcnn(feature, n_class = n_class, image_shape = image_shape, n_feature = n_feature, n_depth = n_depth, sub_sampling = sub_sampling,
               scale = scale, ratio = ratio, auto_scale = auto_scale,
               mask = mask, cascade = cascade, mask_info_flow = mask_info_flow, semantic_feature = semantic_feature,
               proposal_count = proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, valid = valid, performance_count = performance_count,
               pool_size = pool_size, method = method,
               mean = mean, std = std, clip_ratio = clip_ratio, batch_size = batch_size,
               sub_n_feature = sub_n_feature, sub_momentum = sub_momentum, sub_epsilon = sub_epsilon, fpn = fpn, fpn_n_depth = fpn_n_depth,
               rpn_feature_share = rpn_feature_share, rpn_n_feature = rpn_n_feature, rpn_use_bias = rpn_use_bias, rpn_activation = rpn_activation,
               cls_n_feature = cls_n_feature, cls_activation = cls_activation,
               mask_n_feature = mask_n_feature, mask_n_depth = mask_n_depth, mask_activation = mask_activation,
               semantic_level = semantic_level, semantic_n_feature = semantic_n_feature, semantic_n_depth = semantic_n_depth, semantic_pool_size = semantic_pool_size, semantic_activation = semantic_activation,
               sampling_count = sampling_count, sampling_positive_ratio = sampling_positive_ratio, sampling_positive_threshold = sampling_positive_threshold, sampling_negative_threshold = sampling_negative_threshold)
    return out

def cascade_rcnn(feature, n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4, sub_sampling = 1,
                 scale = [0.03125, 0.0625, 0.125, 0.25, 0.5], ratio = [0.5, 1, 2], auto_scale = True,
                 mask = False, cascade = True, mask_info_flow = False, semantic_feature = False,
                 proposal_count = 1000, iou_threshold = 0.7, soft_nms = True, valid = True, performance_count = 5000,
                 pool_size = 7, method = "bilinear",
                 mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, batch_size = 1,
                 sub_n_feature = None, sub_momentum = 0.997, sub_epsilon = 1e-4, fpn = fpn, fpn_n_depth = 1,
                 rpn_feature_share = True, rpn_n_feature = None, rpn_use_bias = False, rpn_activation = tf.keras.activations.relu,
                 cls_n_feature = None, cls_activation = tf.keras.activations.relu,
                 mask_n_feature = None, mask_n_depth = 4, mask_activation = tf.keras.activations.relu,
                 semantic_level = 1, semantic_n_feature = None, semantic_n_depth = 4, semantic_pool_size = 14, semantic_activation = tf.keras.activations.relu,
                 sampling_count = None, sampling_positive_ratio = 0.25, sampling_positive_threshold = 0.5, sampling_negative_threshold = 0.5):
    out = rcnn(feature, n_class = n_class, image_shape = image_shape, n_feature = n_feature, n_depth = n_depth, sub_sampling = sub_sampling,
               scale = scale, ratio = ratio, auto_scale = auto_scale,
               mask = mask, cascade = cascade, mask_info_flow = mask_info_flow, semantic_feature = semantic_feature,
               proposal_count = proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, valid = valid, performance_count = performance_count,
               pool_size = pool_size, method = method,
               mean = mean, std = std, clip_ratio = clip_ratio, batch_size = batch_size,
               sub_n_feature = sub_n_feature, sub_momentum = sub_momentum, sub_epsilon = sub_epsilon, fpn = fpn, fpn_n_depth = fpn_n_depth,
               rpn_feature_share = rpn_feature_share, rpn_n_feature = rpn_n_feature, rpn_use_bias = rpn_use_bias, rpn_activation = rpn_activation,
               cls_n_feature = cls_n_feature, cls_activation = cls_activation,
               mask_n_feature = mask_n_feature, mask_n_depth = mask_n_depth, mask_activation = mask_activation,
               semantic_level = semantic_level, semantic_n_feature = semantic_n_feature, semantic_n_depth = semantic_n_depth, semantic_pool_size = semantic_pool_size, semantic_activation = semantic_activation,
               sampling_count = sampling_count, sampling_positive_ratio = sampling_positive_ratio, sampling_positive_threshold = sampling_positive_threshold, sampling_negative_threshold = sampling_negative_threshold)
    return out

def hybrid_task_cascade_rcnn(feature, n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4, sub_sampling = 1,
                             scale = [0.03125, 0.0625, 0.125, 0.25, 0.5], ratio = [0.5, 1, 2], auto_scale = True,
                             mask = True, cascade = True, mask_info_flow = True, semantic_feature = True,
                             proposal_count = 1000, iou_threshold = 0.7, soft_nms = True, valid = True, performance_count = 5000,
                             pool_size = 7, method = "bilinear",
                             mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, batch_size = 1,
                             sub_n_feature = None, sub_momentum = 0.997, sub_epsilon = 1e-4, fpn = fpn, fpn_n_depth = 1,
                             rpn_feature_share = True, rpn_n_feature = None, rpn_use_bias = False, rpn_activation = tf.keras.activations.relu,
                             cls_n_feature = None, cls_activation = tf.keras.activations.relu,
                             mask_n_feature = None, mask_n_depth = 4, mask_activation = tf.keras.activations.relu,
                             semantic_level = 1, semantic_n_feature = None, semantic_n_depth = 4, semantic_pool_size = 14, semantic_activation = tf.keras.activations.relu,
                             sampling_count = None, sampling_positive_ratio = 0.25, sampling_positive_threshold = 0.5, sampling_negative_threshold = 0.5):
    out = rcnn(feature, n_class = n_class, image_shape = image_shape, n_feature = n_feature, n_depth = n_depth, sub_sampling = sub_sampling,
               scale = scale, ratio = ratio, auto_scale = auto_scale,
               mask = mask, cascade = cascade, mask_info_flow = mask_info_flow, semantic_feature = semantic_feature,
               proposal_count = proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, valid = valid, performance_count = performance_count,
               pool_size = pool_size, method = method,
               mean = mean, std = std, clip_ratio = clip_ratio, batch_size = batch_size,
               sub_n_feature = sub_n_feature, sub_momentum = sub_momentum, sub_epsilon = sub_epsilon, fpn = fpn, fpn_n_depth = fpn_n_depth,
               rpn_feature_share = rpn_feature_share, rpn_n_feature = rpn_n_feature, rpn_use_bias = rpn_use_bias, rpn_activation = rpn_activation,
               cls_n_feature = cls_n_feature, cls_activation = cls_activation,
               mask_n_feature = mask_n_feature, mask_n_depth = mask_n_depth, mask_activation = mask_activation,
               semantic_level = semantic_level, semantic_n_feature = semantic_n_feature, semantic_n_depth = semantic_n_depth, semantic_pool_size = semantic_pool_size, semantic_activation = semantic_activation,
               sampling_count = sampling_count, sampling_positive_ratio = sampling_positive_ratio, sampling_positive_threshold = sampling_positive_threshold, sampling_negative_threshold = sampling_negative_threshold)
    return out
