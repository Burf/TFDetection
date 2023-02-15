import tensorflow as tf
import functools
import numpy as np

from tfdet.core.assign import max_iou, random_sampler
from tfdet.core.util import map_fn
from ..head import rpn_head, bbox_head, mask_head, semantic_head, Rpn2Proposal, Classifier2Proposal
from ..neck import FeatureAlign, fpn
from ..train.loss import RoiTarget

def neck_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "glorot_uniform", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def rpn_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def roi_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def neck(n_feature = 256, n_sampling = 1, pre_sampling = False, neck = fpn, neck_n_depth = 1, use_bias = None, convolution = neck_conv, normalize = None, **kwargs):
    return FeatureAlign(n_feature = n_feature, n_sampling = n_sampling, pre_sampling = pre_sampling, neck = neck, neck_n_depth = neck_n_depth, use_bias = use_bias, convolution = convolution, normalize = normalize, **kwargs)

roi_assign = [functools.partial(max_iou, positive_threshold = threshold, negative_threshold = threshold, min_threshold = threshold, match_low_quality = False, mode = "normal") for threshold in [0.5, 0.6, 0.7]]

def roi_sampler(true_indices, positive_indices, negative_indices, sampling_count = 512, positive_ratio = 0.25, return_count = False):
    return random_sampler(true_indices, positive_indices, negative_indices, sampling_count = sampling_count, positive_ratio = positive_ratio, return_count = return_count)

def rcnn(feature, neck = neck, rpn_head = rpn_head, bbox_head = bbox_head, mask_head = None, semantic_head = None,
         cascade = False, interleaved = False, mask_info_flow = False,
         proposal_count = 1000, iou_threshold = 0.7, score_threshold = float('-inf'), soft_nms = False, valid_inside_anchor = False, performance_count = 5000,
         rpn_mean = [0., 0., 0., 0.], rpn_std = [1., 1., 1., 1.], rpn_clip_ratio = 16 / 1000, 
         cls_mean = [0., 0., 0., 0.], cls_std = [0.1, 0.1, 0.2, 0.2], cls_clip_ratio = 16 / 1000,
         batch_size = 1,
         train = False, assign = roi_assign, sampler = [roi_sampler, roi_sampler, roi_sampler], mask_size = 28, method = "bilinear", add_gt_in_sampler = True):
    """
    train = True (for training to apply prior to roi sampling)> return rpn_y_pred, rpn_bbox_pred, semantic_pred(optional), train_tag(train_tag is a argument for train_model)
    """
    if not isinstance(feature, (tuple, list)):
        feature = [feature]
    if not isinstance(assign, (tuple, list)):
        assign = [assign]
    if not isinstance(sampler, (tuple, list)):
        sampler = [sampler]
    feature = list(feature)
    feature_count = len(feature)
    assign = assign * 3 if cascade and len(assign) == 1 else assign
    sampler = sampler * 3 if cascade and len(sampler) == 1 else sampler
    
    if neck is not None:
        feature = neck(name = "neck")(feature)
    
    rpn_y_pred, rpn_bbox_pred, anchors = rpn_head(feature)
    proposals = Rpn2Proposal(proposal_count, iou_threshold = iou_threshold, soft_nms = soft_nms, valid_inside_anchor = valid_inside_anchor, performance_count = performance_count,
                             mean = rpn_mean, std = rpn_std, clip_ratio = rpn_clip_ratio,
                             batch_size = batch_size, dtype = tf.float32, name = "proposals")([rpn_y_pred, rpn_bbox_pred, anchors])
    
    train_tag = None
    if train:
        y_true = tf.keras.layers.Input(shape = (None, None), name = "y_true") #(batch_size, padded_num_true, 1 or n_class)
        bbox_true = tf.keras.layers.Input(shape = (None, 4), name = "bbox_true") #(batch_size, padded_num_true, 4)
        mask_true = tf.keras.layers.Input(shape = (None, None, None, 1), dtype = tf.uint8, name = "mask_true") if mask_head is not None or semantic_head is not None else None #(batch_size, padded_num_true, h, w, 1)
        train_tag = {"inputs":[arg for arg in [y_true, bbox_true, mask_true] if arg is not None],
                     "bbox_inputs":[],
                     "bbox_outputs":[],
                     "mask_inputs":[],
                     "mask_outputs":[],
                     "sampling":sampler[0] is not None}
    
    n_stage = 3 if cascade else 1
    feature = feature[:feature_count]
    mask_feature = semantic_pred = semantic_feature = None
    if semantic_head is not None:
        semantic_pred, semantic_feature = semantic_head(feature)
    cls_y_pred, cls_bbox_pred, proposals, cls_mask_pred = [], [], [proposals], []
    if not (mask_head is not None and interleaved): #normal
        for stage in range(n_stage):
            _proposals = proposals[-1]
            if train:
                target_func = RoiTarget(assign = assign[stage], sampler = sampler[stage], mask_size = mask_size, method = method, add_gt_in_sampler = add_gt_in_sampler, name = "roi_target_{0}".format(stage + 1))
                out = target_func([arg for arg in [y_true, bbox_true, mask_true] if arg is not None], _proposals)
                target_inputs = out[:3] #state, y_true, bbox_true, mask_true(optional) for target
                proposals[-1] = _proposals = out[-1]
                if mask_head is not None:
                    target_mask_inputs = out[:4]

            _cls_y_pred, _cls_bbox_pred = bbox_head(feature, _proposals, semantic_feature = semantic_feature)
            cls_y_pred.append(_cls_y_pred)
            cls_bbox_pred.append(_cls_bbox_pred)
                
            if mask_head is not None:
                _cls_mask_pred, mask_feature = mask_head(feature, _proposals, mask_feature = mask_feature if mask_info_flow else None, semantic_feature = semantic_feature)
                cls_mask_pred.append(_cls_mask_pred)
                
            if train:
                train_tag["bbox_inputs"].append(target_inputs)
                train_tag["bbox_outputs"].append([_cls_y_pred, _cls_bbox_pred, _proposals])
                if mask_head is not None:
                    train_tag["mask_inputs"].append(target_mask_inputs)
                    train_tag["mask_outputs"].append(_cls_mask_pred)

            if stage + 1 < n_stage:
                _proposals = Classifier2Proposal(mean = cls_mean, std = np.divide(cls_std, stage + 1), clip_ratio = cls_clip_ratio,
                                                 dtype = tf.float32, name = "proposals_{0}".format(stage + 2))([_cls_y_pred, _cls_bbox_pred, _proposals])
                proposals.append(_proposals)
    else: #mask_head + interleaved
        for stage in range(n_stage):
            _proposals = proposals[-1]
            if train:
                target_func = RoiTarget(assign = assign[stage], sampler = sampler[stage], add_gt_in_sampler = add_gt_in_sampler, name = "roi_bbox_target_{0}".format(stage + 1))
                out = target_func([y_true, bbox_true], _proposals)
                target_inputs = out[:3] #state, y_true, bbox_true for target
                proposals[-1] = _proposals = out[-1]

            _cls_y_pred, _cls_bbox_pred = bbox_head(feature, _proposals, semantic_feature = semantic_feature)
            cls_y_pred.append(_cls_y_pred)
            cls_bbox_pred.append(_cls_bbox_pred)
            
            if train:
                train_tag["bbox_inputs"].append(target_inputs)
                train_tag["bbox_outputs"].append([_cls_y_pred, _cls_bbox_pred, _proposals])
            
            _proposals = Classifier2Proposal(mean = cls_mean, std = np.divide(cls_std, stage + 1), clip_ratio = cls_clip_ratio,
                                             dtype = tf.float32, name = "proposals_{0}".format(stage + 2))([_cls_y_pred, _cls_bbox_pred, _proposals])
            proposals.append(_proposals)
            
            if train:
                target_func = RoiTarget(assign = assign[stage], sampler = sampler[stage], mask_size = mask_size, method = method, add_gt_in_sampler = add_gt_in_sampler, name = "roi_mask_target_{0}".format(stage + 1))
                out = target_func([y_true, bbox_true, mask_true], _proposals)
                target_inputs = out[:4] #state, y_true, bbox_true, mask_true for target
                _proposals = out[-1]

            _cls_mask_pred, mask_feature = mask_head(feature, _proposals, mask_feature = mask_feature if mask_info_flow else None, semantic_feature = semantic_feature)
            cls_mask_pred.append(_cls_mask_pred)

            if train:
                train_tag["mask_inputs"].append(target_inputs)
                train_tag["mask_outputs"].append(_cls_mask_pred)
    
    if len(cls_mask_pred) == 0:
        cls_mask_pred = None
    cls_y_pred, cls_bbox_pred, proposals, cls_mask_pred = [r[0] if r is not None and len(r) == 1 else r for r in [cls_y_pred, cls_bbox_pred, proposals, cls_mask_pred]]
    result = [r for r in [rpn_y_pred, rpn_bbox_pred, anchors, cls_y_pred, cls_bbox_pred, proposals, cls_mask_pred, semantic_pred, train_tag] if r is not None]
    return result

def faster_rcnn(feature, n_class = 21, image_shape = [1024, 1024], 
                scale = [32, 64, 128, 256, 512], ratio = [0.5, 1, 2], octave = 1,
                rpn_n_feature = 256, rpn_feature_share = True,
                cls_n_feature = 1024, use_bias = None,
                proposal_count = 1000, iou_threshold = 0.7, score_threshold = float('-inf'), soft_nms = False, valid_inside_anchor = False, performance_count = 5000,
                pool_size = 7, method = "bilinear",
                mean = [0., 0., 0., 0.], std = [1., 1., 1., 1.], clip_ratio = 16 / 1000, 
                batch_size = 1, add_gt_in_sampler = True,
                train = False, assign = roi_assign[0], sampler = roi_sampler,
                neck = neck,
                rpn_convolution = rpn_conv, rpn_normalize = None, rpn_activation = tf.keras.activations.relu,
                cls_convolution = roi_conv, cls_normalize = None, cls_activation = tf.keras.activations.relu):
    """
    train = True (for training to apply prior to roi sampling)> return rpn_y_pred, rpn_bbox_pred, semantic_pred(optional), train_tag(train_tag is a argument for train_model)
    """
    _rpn_head = functools.partial(rpn_head, image_shape = image_shape,
                                  scale = scale, ratio = ratio, octave = octave,
                                  n_feature = rpn_n_feature, use_bias = use_bias, feature_share = rpn_feature_share,
                                  convolution = rpn_convolution, normalize = rpn_normalize, activation = rpn_activation)
    _bbox_head = functools.partial(bbox_head, n_class = n_class, image_shape = image_shape, n_feature = cls_n_feature, use_bias = use_bias,
                                   pool_size = pool_size, method = method,
                                   convolution = cls_convolution, normalize = cls_normalize, activation = cls_activation)
    return rcnn(feature, neck = neck, rpn_head = _rpn_head, bbox_head = _bbox_head,
                cascade = False, interleaved = False, mask_info_flow = False,
                proposal_count = proposal_count, iou_threshold = iou_threshold, score_threshold = score_threshold, soft_nms = soft_nms, valid_inside_anchor = valid_inside_anchor, performance_count = performance_count,
                rpn_mean = mean, rpn_std = std, rpn_clip_ratio = clip_ratio, batch_size = batch_size,
                train = train, assign = assign, sampler = sampler, add_gt_in_sampler = add_gt_in_sampler)

def mask_rcnn(feature, n_class = 21, image_shape = [1024, 1024], interleaved = False,
              scale = [32, 64, 128, 256, 512], ratio = [0.5, 1, 2], octave = 1,
              rpn_n_feature = 256, rpn_feature_share = True,
              cls_n_feature = 1024, mask_n_feature = 256, mask_n_depth = 4, mask_scale = 2, use_bias = None,
              proposal_count = 1000, iou_threshold = 0.7, score_threshold = float('-inf'), soft_nms = False, valid_inside_anchor = False, performance_count = 5000,
              pool_size = 7, mask_pool_size = 14, method = "bilinear",
              rpn_mean = [0., 0., 0., 0.], rpn_std = [1., 1., 1., 1.], rpn_clip_ratio = 16 / 1000, 
              cls_mean = [0., 0., 0., 0.], cls_std = [0.1, 0.1, 0.2, 0.2], cls_clip_ratio = 16 / 1000, add_gt_in_sampler = True,
              batch_size = 1,
              train = False, assign = roi_assign[0], sampler = roi_sampler,
              neck = neck,
              rpn_convolution = rpn_conv, rpn_normalize = None, rpn_activation = tf.keras.activations.relu,
              cls_convolution = roi_conv, cls_normalize = None, cls_activation = tf.keras.activations.relu,
              mask_convolution = roi_conv, mask_normalize = None, mask_activation = tf.keras.activations.relu):
    """
    train = True (for training to apply prior to roi sampling)> return rpn_y_pred, rpn_bbox_pred, semantic_pred(optional), train_tag(train_tag is a argument for train_model)
    """
    _rpn_head = functools.partial(rpn_head, image_shape = image_shape,
                                  scale = scale, ratio = ratio, octave = octave,
                                  n_feature = rpn_n_feature, use_bias = use_bias, feature_share = rpn_feature_share,
                                  convolution = rpn_convolution, normalize = rpn_normalize, activation = rpn_activation)
    _bbox_head = functools.partial(bbox_head, n_class = n_class, image_shape = image_shape, n_feature = cls_n_feature, use_bias = use_bias,
                                   pool_size = pool_size, method = method,
                                   convolution = cls_convolution, normalize = cls_normalize, activation = cls_activation)
    _mask_head = functools.partial(mask_head, n_class = n_class, image_shape = image_shape, n_feature = mask_n_feature, n_depth = mask_n_depth, scale = mask_scale, use_bias = use_bias,
                                   pool_size = mask_pool_size, method = method,
                                   convolution = mask_convolution, normalize = mask_normalize, activation = mask_activation)
    return rcnn(feature, neck = neck, rpn_head = _rpn_head, bbox_head = _bbox_head, mask_head = _mask_head,
                cascade = False, interleaved = interleaved, mask_info_flow = False,
                proposal_count = proposal_count, iou_threshold = iou_threshold, score_threshold = score_threshold, soft_nms = soft_nms, valid_inside_anchor = valid_inside_anchor, performance_count = performance_count,
                rpn_mean = rpn_mean, rpn_std = rpn_std, rpn_clip_ratio = rpn_clip_ratio, 
                cls_mean = cls_mean, cls_std = cls_std, cls_clip_ratio = cls_clip_ratio, 
                batch_size = batch_size,
                train = train, assign = assign, sampler = sampler, mask_size = mask_pool_size * 2, method = method, add_gt_in_sampler = add_gt_in_sampler)

def cascade_rcnn(feature, n_class = 21, image_shape = [1024, 1024], mask = False, interleaved = False, mask_info_flow = False,
                 scale = [32, 64, 128, 256, 512], ratio = [0.5, 1, 2], octave = 1,
                 rpn_n_feature = 256, rpn_feature_share = True,
                 cls_n_feature = 1024, mask_n_feature = 256, mask_n_depth = 4, mask_scale = 2, use_bias = None,
                 proposal_count = 1000, iou_threshold = 0.7, score_threshold = float('-inf'), soft_nms = False, valid_inside_anchor = False, performance_count = 5000,
                 pool_size = 7, mask_pool_size = 14, method = "bilinear",
                 rpn_mean = [0., 0., 0., 0.], rpn_std = [1., 1., 1., 1.], rpn_clip_ratio = 16 / 1000, 
                 cls_mean = [0., 0., 0., 0.], cls_std = [0.1, 0.1, 0.2, 0.2], cls_clip_ratio = 16 / 1000,
                 batch_size = 1,
                 train = False, assign = roi_assign, sampler = [roi_sampler, roi_sampler, roi_sampler], add_gt_in_sampler = True,
                 neck = neck,
                 rpn_convolution = rpn_conv, rpn_normalize = None, rpn_activation = tf.keras.activations.relu,
                 cls_convolution = roi_conv, cls_normalize = None, cls_activation = tf.keras.activations.relu,
                 mask_convolution = roi_conv, mask_normalize = None, mask_activation = tf.keras.activations.relu):
    """
    train = True (for training to apply prior to roi sampling)> return rpn_y_pred, rpn_bbox_pred, semantic_pred(optional), train_tag(train_tag is a argument for train_model)
    """
    _rpn_head = functools.partial(rpn_head, image_shape = image_shape,
                                  scale = scale, ratio = ratio, octave = octave,
                                  n_feature = rpn_n_feature, use_bias = use_bias, feature_share = rpn_feature_share,
                                  convolution = rpn_convolution, normalize = rpn_normalize, activation = rpn_activation)
    _bbox_head = functools.partial(bbox_head, n_class = n_class, image_shape = image_shape, n_feature = cls_n_feature, use_bias = use_bias,
                                   pool_size = pool_size, method = method,
                                   convolution = cls_convolution, normalize = cls_normalize, activation = cls_activation)
    _mask_head = None
    if mask:
        _mask_head = functools.partial(mask_head, n_class = n_class, image_shape = image_shape, n_feature = mask_n_feature, n_depth = mask_n_depth, scale = mask_scale, use_bias = use_bias,
                                       pool_size = mask_pool_size, method = method,
                                       convolution = mask_convolution, normalize = mask_normalize, activation = mask_activation)
    return rcnn(feature, neck = neck, rpn_head = _rpn_head, bbox_head = _bbox_head, mask_head = _mask_head,
                cascade = True, interleaved = interleaved, mask_info_flow = mask_info_flow,
                proposal_count = proposal_count, iou_threshold = iou_threshold, score_threshold = score_threshold, soft_nms = soft_nms, valid_inside_anchor = valid_inside_anchor, performance_count = performance_count,
                rpn_mean = rpn_mean, rpn_std = rpn_std, rpn_clip_ratio = rpn_clip_ratio, 
                cls_mean = cls_mean, cls_std = cls_std, cls_clip_ratio = cls_clip_ratio, 
                batch_size = batch_size,
                train = train, assign = assign, sampler = sampler, mask_size = mask_pool_size * 2, method = method, add_gt_in_sampler = add_gt_in_sampler)
    
def hybrid_task_cascade_rcnn(feature, n_class = 21, image_shape = [1024, 1024], mask = True, interleaved = True, mask_info_flow = True, semantic_feature = True,
                             scale = [32, 64, 128, 256, 512], ratio = [0.5, 1, 2], octave = 1,
                             rpn_n_feature = 256, rpn_feature_share = True,
                             cls_n_feature = 1024, mask_n_feature = 256, mask_n_depth = 4, mask_scale = 2, semantic_level = 1, semantic_n_feature = 256, semantic_n_depth = 4, use_bias = None,
                             proposal_count = 1000, iou_threshold = 0.7, score_threshold = float('-inf'), soft_nms = False, valid_inside_anchor = False, performance_count = 5000,
                             pool_size = 7, mask_pool_size = 14, method = "bilinear",
                             rpn_mean = [0., 0., 0., 0.], rpn_std = [1., 1., 1., 1.], rpn_clip_ratio = 16 / 1000, 
                             cls_mean = [0., 0., 0., 0.], cls_std = [0.1, 0.1, 0.2, 0.2], cls_clip_ratio = 16 / 1000,
                             batch_size = 1,
                             train = False, assign = roi_assign, sampler = [roi_sampler, roi_sampler, roi_sampler], add_gt_in_sampler = True,
                             neck = neck,
                             rpn_convolution = rpn_conv, rpn_normalize = None, rpn_activation = tf.keras.activations.relu,
                             cls_convolution = roi_conv, cls_normalize = None, cls_activation = tf.keras.activations.relu,
                             mask_convolution = roi_conv, mask_normalize = None, mask_activation = tf.keras.activations.relu,
                             semantic_logits_activation = tf.keras.activations.softmax, semantic_convolution = roi_conv, semantic_normalize = None, semantic_activation = tf.keras.activations.relu):
    """
    train = True (for training to apply prior to roi sampling)> return rpn_y_pred, rpn_bbox_pred, semantic_pred(optional), train_tag(train_tag is a argument for train_model)
    """
    _rpn_head = functools.partial(rpn_head, image_shape = image_shape,
                                  scale = scale, ratio = ratio, octave = octave,
                                  n_feature = rpn_n_feature, use_bias = use_bias, feature_share = rpn_feature_share,
                                  convolution = rpn_convolution, normalize = rpn_normalize, activation = rpn_activation)
    _bbox_head = functools.partial(bbox_head, n_class = n_class, image_shape = image_shape, n_feature = cls_n_feature, use_bias = use_bias,
                                   pool_size = pool_size, method = method,
                                   convolution = cls_convolution, normalize = cls_normalize, activation = cls_activation)
    _semantic_head = _mask_head = None
    if mask:
        _mask_head = functools.partial(mask_head, n_class = n_class, image_shape = image_shape, n_feature = mask_n_feature, n_depth = mask_n_depth, scale = mask_scale, use_bias = use_bias,
                                       pool_size = mask_pool_size, method = method,
                                       convolution = mask_convolution, normalize = mask_normalize, activation = mask_activation)
    if semantic_feature:
        _semantic_head = functools.partial(semantic_head, n_class = n_class, level = semantic_level, n_feature = semantic_n_feature, n_depth = semantic_n_depth, use_bias = use_bias, method = method, logits_activation = semantic_logits_activation, convolution = semantic_convolution, normalize = semantic_normalize, activation = semantic_activation)
    return rcnn(feature, neck = neck, rpn_head = _rpn_head, bbox_head = _bbox_head, mask_head = _mask_head, semantic_head = _semantic_head,
                cascade = True, interleaved = interleaved, mask_info_flow = mask_info_flow,
                proposal_count = proposal_count, iou_threshold = iou_threshold, score_threshold = score_threshold, soft_nms = soft_nms, valid_inside_anchor = valid_inside_anchor, performance_count = performance_count,
                rpn_mean = rpn_mean, rpn_std = rpn_std, rpn_clip_ratio = rpn_clip_ratio, 
                cls_mean = cls_mean, cls_std = cls_std, cls_clip_ratio = cls_clip_ratio, 
                batch_size = batch_size,
                train = train, assign = assign, sampler = sampler, mask_size = mask_pool_size * 2, method = method, add_gt_in_sampler = add_gt_in_sampler)