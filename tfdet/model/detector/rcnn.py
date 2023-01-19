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

def assign(y_true, bbox_true, y_pred, bbox_pred, positive_threshold = 0.5, negative_threshold = 0.5, min_threshold = 0.5, match_low_quality = False, mode = "normal"):
    return max_iou(y_true, bbox_true, y_pred, bbox_pred, positive_threshold = positive_threshold, negative_threshold = negative_threshold, min_threshold = min_threshold, match_low_quality = match_low_quality, mode = mode)

def assign2(y_true, bbox_true, y_pred, bbox_pred, positive_threshold = 0.6, negative_threshold = 0.6, min_threshold = 0.6, match_low_quality = False, mode = "normal"):
    return max_iou(y_true, bbox_true, y_pred, bbox_pred, positive_threshold = positive_threshold, negative_threshold = negative_threshold, min_threshold = min_threshold, match_low_quality = match_low_quality, mode = mode)

def assign3(y_true, bbox_true, y_pred, bbox_pred, positive_threshold = 0.7, negative_threshold = 0.7, min_threshold = 0.7, match_low_quality = False, mode = "normal"):
    return max_iou(y_true, bbox_true, y_pred, bbox_pred, positive_threshold = positive_threshold, negative_threshold = negative_threshold, min_threshold = min_threshold, match_low_quality = match_low_quality, mode = mode)

def rcnn(feature, neck = neck, rpn_head = rpn_head, bbox_head = bbox_head, mask_head = None, semantic_head = None,
         cascade = False, interleaved = False, mask_info_flow = False,
         proposal_count = 1000, iou_threshold = 0.7, soft_nms = False, valid = False, performance_count = 5000,
         rpn_mean = [0., 0., 0., 0.], rpn_std = [1., 1., 1., 1.], rpn_clip_ratio = 16 / 1000, 
         cls_mean = [0., 0., 0., 0.], cls_std = [0.1, 0.1, 0.2, 0.2], cls_clip_ratio = 16 / 1000,
         batch_size = 1,
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
    proposals = Rpn2Proposal(proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, valid = valid, performance_count = performance_count, batch_size = batch_size, mean = rpn_mean, std = rpn_std, clip_ratio = rpn_clip_ratio, dtype = tf.float32, name = "proposals")([rpn_score, rpn_regress], anchors)
    
    sampling_tag = None
    if isinstance(sampling_count, int) and 0 < sampling_count:
        sampling_y_true = y_true = tf.keras.layers.Input(shape = (None, None), name = "y_true") #(batch_size, padded_num_true, 1 or n_class)
        sampling_bbox_true = bbox_true = tf.keras.layers.Input(shape = (None, 4), name = "bbox_true") #(batch_size, padded_num_true, 4)
        sampling_mask_true = mask_true = tf.keras.layers.Input(shape = (None, None, None, 1), name = "mask_true") if mask_head is not None or semantic_head is not None else None #(batch_size, padded_num_true, h, w)
        
        sampling_func = lambda args, **kwarg: map_fn(sampling_target, *args, dtype = (tf.float32, tf.float32, tf.float32), batch_size = batch_size, **kwargs)
        sampling_mask_func = lambda args, **kwarg: map_fn(sampling_target, *args, dtype = (tf.float32, tf.float32, tf.float32, tf.float32), batch_size = batch_size, **kwargs)
        sampling_tag = {"sampling_assign":sampling_assign, "sampling_count":sampling_count, "positive_ratio":sampling_positive_ratio, 
                        "y_true":y_true, "bbox_true":bbox_true, "mask_true":mask_true, "sampling_y_true":[], "sampling_bbox_true":[], "sampling_mask_true":[]}
    
    n_stage = 3 if cascade else 1
    feature = feature[:feature_count]
    mask_feature = semantic_regress = semantic_feature = None
    if semantic_head is not None:
        semantic_regress, semantic_feature = semantic_head(feature)
    cls_logits, cls_regress, proposals, mask_regress = [], [], [proposals], []
    for stage in range(n_stage):
        _proposals = proposals[-1]
        if sampling_tag is not None:
            kwargs = {"assign":sampling_assign[stage if stage < len(sampling_assign) else stage - 1], "sampling_count":sampling_count, "positive_ratio":sampling_positive_ratio}
            if mask_head:
                sampling_y_true, sampling_bbox_true, sampling_mask_true, _proposals = tf.keras.layers.Lambda(sampling_mask_func, arguments = kwargs, dtype = tf.float32, name = "sampling_target_{0}".format(stage + 1))([sampling_y_true, sampling_bbox_true, _proposals, sampling_mask_true])
            else:
                sampling_y_true, sampling_bbox_true, _proposals = tf.keras.layers.Lambda(sampling_func, arguments = kwargs, dtype = tf.float32, name = "sampling_target_{0}".format(stage + 1))([sampling_y_true, sampling_bbox_true, _proposals])
            sampling_tag["sampling_y_true"].append(sampling_y_true)
            sampling_tag["sampling_bbox_true"].append(sampling_bbox_true)
            sampling_tag["sampling_mask_true"].append(sampling_mask_true)
            proposals[-1] = _proposals
        
        _cls_logits, _cls_regress = bbox_head(feature, _proposals, semantic_feature = semantic_feature)
        cls_logits.append(_cls_logits)
        cls_regress.append(_cls_regress)
        
        if (stage + 1) < (n_stage + int(mask_head is not None and interleaved)):
            out = Classifier2Proposal(batch_size, mean = cls_mean, std = np.divide(cls_std, stage + 1), clip_ratio = cls_clip_ratio, dtype = tf.float32, name = "proposals_{0}".format(stage + 2))([cls_logits[-1], cls_regress[-1], proposals[-1]])
            if interleaved:
                _proposals = out
            proposals.append(out)
            
        if mask_head is not None:
            _mask_regress, mask_feature = mask_head(feature, _proposals, mask_feature = mask_feature if mask_info_flow else None, semantic_feature = semantic_feature)
            mask_regress.append(_mask_regress)
    
    if len(mask_regress) == 0:
        mask_regress = None
    cls_logits, cls_regress, proposals, mask_regress = [r[0] if r is not None and len(r) == 1 else r for r in [cls_logits, cls_regress, proposals, mask_regress]]
    result = [r for r in [rpn_score, rpn_regress, anchors, cls_logits, cls_regress, proposals, mask_regress, semantic_regress, sampling_tag] if r is not None]
    return result

def faster_rcnn(feature, n_class = 21, image_shape = [1024, 1024], 
                scale = [32, 64, 128, 256, 512], ratio = [0.5, 1, 2], octave = 1,
                rpn_n_feature = 256, rpn_use_bias = True, rpn_feature_share = True,
                cls_n_feature = 1024,
                proposal_count = 1000, iou_threshold = 0.7, soft_nms = False, valid = False, performance_count = 5000,
                pool_size = 7, method = "bilinear",
                mean = [0., 0., 0., 0.], std = [1., 1., 1., 1.], clip_ratio = 16 / 1000, batch_size = 1,
                sampling_assign = assign, sampling_count = None, sampling_positive_ratio = 0.25,
                neck = neck,
                rpn_convolution = conv, rpn_normalize = None, rpn_activation = tf.keras.activations.relu,
                cls_convolution = conv, cls_normalize = tf.keras.layers.BatchNormalization, cls_activation = tf.keras.activations.relu):
    """
    for speed training(with roi sampling) > sampling_count = 256 (Recommended value for training to apply prior to roi sampling)> return rcnn_layers(rpn, cls, ...) + sampling_tag(sampling_tag is a argument for rcnn_train_model)
    """
    _rpn_head = functools.partial(rpn_head, image_shape = image_shape,
                                  scale = scale, ratio = ratio, octave = octave,
                                  n_feature = rpn_n_feature, use_bias = rpn_use_bias, feature_share = rpn_feature_share,
                                  convolution = rpn_convolution, normalize = rpn_normalize, activation = rpn_activation)
    _bbox_head = functools.partial(bbox_head, n_class = n_class, image_shape = image_shape, n_feature = cls_n_feature,
                                   pool_size = pool_size, method = method,
                                   convolution = cls_convolution, normalize = cls_normalize, activation = cls_activation)
    return rcnn(feature, neck = neck, rpn_head = _rpn_head, bbox_head = _bbox_head,
                cascade = False, interleaved = False, mask_info_flow = False,
                proposal_count = proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, valid = valid, performance_count = performance_count,
                rpn_mean = mean, rpn_std = std, rpn_clip_ratio = clip_ratio, batch_size = batch_size,
                sampling_assign = sampling_assign, sampling_count = sampling_count, sampling_positive_ratio = sampling_positive_ratio)

def mask_rcnn(feature, n_class = 21, image_shape = [1024, 1024], interleaved = False,
              scale = [32, 64, 128, 256, 512], ratio = [0.5, 1, 2], octave = 1,
              rpn_n_feature = 256, rpn_use_bias = True, rpn_feature_share = True,
              cls_n_feature = 1024, mask_n_feature = 256, mask_n_depth = 4, 
              proposal_count = 1000, iou_threshold = 0.7, soft_nms = False, valid = False, performance_count = 5000,
              pool_size = 7, method = "bilinear",
              rpn_mean = [0., 0., 0., 0.], rpn_std = [1., 1., 1., 1.], rpn_clip_ratio = 16 / 1000, 
              cls_mean = [0., 0., 0., 0.], cls_std = [0.1, 0.1, 0.2, 0.2], cls_clip_ratio = 16 / 1000,
              batch_size = 1,
              sampling_assign = assign, sampling_count = None, sampling_positive_ratio = 0.25,
              neck = neck,
              rpn_convolution = conv, rpn_normalize = None, rpn_activation = tf.keras.activations.relu,
              cls_convolution = conv, cls_normalize = tf.keras.layers.BatchNormalization, cls_activation = tf.keras.activations.relu,
              mask_convolution = conv, mask_normalize = tf.keras.layers.BatchNormalization, mask_activation = tf.keras.activations.relu):
    """
    for speed training(with roi sampling) > sampling_count = 256 (Recommended value for training to apply prior to roi sampling)> return rcnn_layers(rpn, cls, ...) + sampling_tag(sampling_tag is a argument for rcnn_train_model)
    """
    _rpn_head = functools.partial(rpn_head, image_shape = image_shape,
                                  scale = scale, ratio = ratio, octave = octave,
                                  n_feature = rpn_n_feature, use_bias = rpn_use_bias, feature_share = rpn_feature_share,
                                  convolution = rpn_convolution, normalize = rpn_normalize, activation = rpn_activation)
    _bbox_head = functools.partial(bbox_head, n_class = n_class, image_shape = image_shape, n_feature = cls_n_feature, 
                                   pool_size = pool_size, method = method,
                                   convolution = cls_convolution, normalize = cls_normalize, activation = cls_activation)
    _mask_head = functools.partial(mask_head, n_class = n_class, image_shape = image_shape, n_feature = mask_n_feature, n_depth = mask_n_depth,
                                   pool_size = pool_size, method = method,
                                   convolution = mask_convolution, normalize = mask_normalize, activation = mask_activation)
    return rcnn(feature, neck = neck, rpn_head = _rpn_head, bbox_head = _bbox_head, mask_head = _mask_head,
                cascade = False, interleaved = interleaved, mask_info_flow = False,
                proposal_count = proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, valid = valid, performance_count = performance_count,
                rpn_mean = rpn_mean, rpn_std = rpn_std, rpn_clip_ratio = rpn_clip_ratio, 
                cls_mean = cls_mean, cls_std = cls_std, cls_clip_ratio = cls_clip_ratio, 
                batch_size = batch_size,
                sampling_assign = sampling_assign, sampling_count = sampling_count, sampling_positive_ratio = sampling_positive_ratio)

def cascade_rcnn(feature, n_class = 21, image_shape = [1024, 1024], mask = False, interleaved = False, mask_info_flow = False,
                 scale = [32, 64, 128, 256, 512], ratio = [0.5, 1, 2], octave = 1,
                 rpn_n_feature = 256, rpn_use_bias = True, rpn_feature_share = True,
                 cls_n_feature = 1024, mask_n_feature = 256, mask_n_depth = 4, 
                 proposal_count = 1000, iou_threshold = 0.7, soft_nms = False, valid = False, performance_count = 5000,
                 pool_size = 7, method = "bilinear",
                 rpn_mean = [0., 0., 0., 0.], rpn_std = [1., 1., 1., 1.], rpn_clip_ratio = 16 / 1000, 
                 cls_mean = [0., 0., 0., 0.], cls_std = [0.1, 0.1, 0.2, 0.2], cls_clip_ratio = 16 / 1000,
                 batch_size = 1,
                 sampling_assign = [assign, assign2, assign3], sampling_count = None, sampling_positive_ratio = 0.25,
                 neck = neck,
                 rpn_convolution = conv, rpn_normalize = None, rpn_activation = tf.keras.activations.relu,
                 cls_convolution = conv, cls_normalize = tf.keras.layers.BatchNormalization, cls_activation = tf.keras.activations.relu,
                 mask_convolution = conv, mask_normalize = tf.keras.layers.BatchNormalization, mask_activation = tf.keras.activations.relu):
    """
    for speed training(with roi sampling) > sampling_count = 256 (Recommended value for training to apply prior to roi sampling)> return rcnn_layers(rpn, cls, ...) + sampling_tag(sampling_tag is a argument for rcnn_train_model)
    """
    _rpn_head = functools.partial(rpn_head, image_shape = image_shape,
                                  scale = scale, ratio = ratio, octave = octave,
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
                cascade = True, interleaved = interleaved, mask_info_flow = mask_info_flow,
                proposal_count = proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, valid = valid, performance_count = performance_count,
                rpn_mean = rpn_mean, rpn_std = rpn_std, rpn_clip_ratio = rpn_clip_ratio, 
                cls_mean = cls_mean, cls_std = cls_std, cls_clip_ratio = cls_clip_ratio, 
                batch_size = batch_size,
                sampling_assign = sampling_assign, sampling_count = sampling_count, sampling_positive_ratio = sampling_positive_ratio)
    
def hybrid_task_cascade_rcnn(feature, n_class = 21, image_shape = [1024, 1024], mask = True, interleaved = True, mask_info_flow = True, semantic_feature = True,
                             scale = [32, 64, 128, 256, 512], ratio = [0.5, 1, 2], octave = 1,
                             rpn_n_feature = 256, rpn_use_bias = True, rpn_feature_share = True,
                             cls_n_feature = 1024, mask_n_feature = 256, mask_n_depth = 4, semantic_level = 1, semantic_n_feature = 256, semantic_n_depth = 4, 
                             proposal_count = 1000, iou_threshold = 0.7, soft_nms = False, valid = False, performance_count = 5000,
                             pool_size = 7, semantic_pool_size = 14, method = "bilinear",
                             rpn_mean = [0., 0., 0., 0.], rpn_std = [1., 1., 1., 1.], rpn_clip_ratio = 16 / 1000, 
                             cls_mean = [0., 0., 0., 0.], cls_std = [0.1, 0.1, 0.2, 0.2], cls_clip_ratio = 16 / 1000,
                             batch_size = 1,
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
                                  scale = scale, ratio = ratio, octave = octave,
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
                cascade = True, interleaved = interleaved, mask_info_flow = mask_info_flow,
                proposal_count = proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, valid = valid, performance_count = performance_count,
                rpn_mean = rpn_mean, rpn_std = rpn_std, rpn_clip_ratio = rpn_clip_ratio, 
                cls_mean = cls_mean, cls_std = cls_std, cls_clip_ratio = cls_clip_ratio, 
                batch_size = batch_size,
                sampling_assign = sampling_assign, sampling_count = sampling_count, sampling_positive_ratio = sampling_positive_ratio)