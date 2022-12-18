import tensorflow as tf
import numpy as np

from tfdet.core.assign import max_iou
from tfdet.core.loss import regularize as regularize_loss
from tfdet.core.util import map_fn
from .loss.rcnn import score_accuracy, score_loss, logits_accuracy, logits_loss, regress_loss, mask_loss, semantic_loss
from .target import rpn_target, sampling_postprocess, cls_target
from ..postprocess.rcnn import FilterDetection

def rpn_assign(y_true, bbox_true, y_pred, bbox_pred, positive_threshold = 0.7, negative_threshold = 0.3, min_threshold = 0.3, match_low_quality = True, mode = "normal"):
    return max_iou(y_true, bbox_true, y_pred, bbox_pred, positive_threshold = positive_threshold, negative_threshold = negative_threshold, min_threshold = min_threshold, match_low_quality = match_low_quality, mode = mode)

def cls_assign(y_true, bbox_true, y_pred, bbox_pred, positive_threshold = 0.5, negative_threshold = 0.5, min_threshold = 0.5, match_low_quality = False, mode = "normal"):
    return max_iou(y_true, bbox_true, y_pred, bbox_pred, positive_threshold = positive_threshold, negative_threshold = negative_threshold, min_threshold = min_threshold, match_low_quality = match_low_quality, mode = mode)

def cls_assign2(y_true, bbox_true, y_pred, bbox_pred, positive_threshold = 0.6, negative_threshold = 0.6, min_threshold = 0.6, match_low_quality = False, mode = "normal"):
    return max_iou(y_true, bbox_true, y_pred, bbox_pred, positive_threshold = positive_threshold, negative_threshold = negative_threshold, min_threshold = min_threshold, match_low_quality = match_low_quality, mode = mode)

def cls_assign3(y_true, bbox_true, y_pred, bbox_pred, positive_threshold = 0.7, negative_threshold = 0.7, min_threshold = 0.7, match_low_quality = False, mode = "normal"):
    return max_iou(y_true, bbox_true, y_pred, bbox_pred, positive_threshold = positive_threshold, negative_threshold = negative_threshold, min_threshold = min_threshold, match_low_quality = match_low_quality, mode = mode)

def train_model(input, rpn_score = None, rpn_regress = None, anchors = None, cls_logits = None, cls_regress = None, proposals = None, mask_regress = None, semantic_regress = None,
                sampling_tag = None, sampling_count = 256,
                rpn_assign = rpn_assign, rpn_positive_ratio = 0.5,
                cls_assign = [cls_assign, cls_assign2, cls_assign3], cls_positive_ratio = 0.25,
                proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05, soft_nms = False, ensemble = True, valid = False, ignore_label = 0, performance_count = 5000,
                batch_size = 1, 
                rpn_mean = [0., 0., 0., 0.], rpn_std = [1., 1., 1., 1.], rpn_clip_ratio = 16 / 1000, 
                cls_mean = [0., 0., 0., 0.], cls_std = [0.1, 0.1, 0.2, 0.2], cls_clip_ratio = 16 / 1000,
                method = "bilinear",
                regularize = True, weight_decay = 1e-4, focal = True, alpha = 1., gamma = 2., sigma = 1, class_weight = None, stage_weight = [1.0, 0.5, 0.25], semantic_weight = 0.2, threshold = 0.5, missing_value = 0.):
    """
    y_true > #(batch_size, padded_num_true, 1 or n_class)
    bbox_true > #(batch_size, padded_num_true, 4)
    mask_true > #(batch_size, padded_num_true, h, w)
    
    train rpn > train_model(x, rpn_score, rpn_regress, anchors)
    train cls > train_model(x, cls_logits = cls_logits, cls_regress = cls_regress, proposals = proposals)
    train mask > train_model(x, cls_logits = cls_logits, cls_regress = cls_regress, proposals = proposals, mask_regress = mask_regress)
    train semantic context > train_model(x, cls_logits = cls_logits, cls_regress = cls_regress, proposals = proposals, semantic_regress = semantic_regress)
    train total > train_model(x, rpn_score, rpn_regress, anchors, cls_logits, cls_regress, proposals, mask_regress, semantic_regress)
    """
    if isinstance(mask_regress, dict):
        sampling_tag = mask_regress
        mask_regress = None
    elif isinstance(semantic_regress, dict):
        sampling_tag = semantic_regress
        semantic_regress = None
    if mask_regress is not None and semantic_regress is None:
        if not isinstance(mask_regress, list) and tf.keras.backend.ndim(mask_regress) == 4:
            semantic_regress = mask_regress
            mask_regress = None
            
    if cls_logits is not None and cls_regress is not None and proposals is not None:
        args = [cls_logits, cls_regress, proposals]
        if mask_regress is not None:
            args += [mask_regress]
    else:
        args = [rpn_score, rpn_regress, anchors]
        cls_mean, cls_std, cls_clip_ratio = rpn_mean, rpn_std, rpn_clip_ratio
        
    y_true = bbox_true = mask_true = sampling_y_true = sampling_bbox_true = sampling_mask_true = None
    if isinstance(sampling_tag, dict):
        sampling_count, cls_positive_ratio, y_true, bbox_true, mask_true, sampling_y_true, sampling_bbox_true, sampling_mask_true = [sampling_tag[key] for key in ["sampling_count", "positive_ratio", "y_true", "bbox_true", "mask_true", "sampling_y_true", "sampling_bbox_true", "sampling_mask_true"]]
    else:
        bbox_true = tf.keras.layers.Input(shape = (None, 4), name = "bbox_true", dtype = [l[0] if isinstance(l, list) else l for l in [l for l in [rpn_regress, cls_regress] if l is not None]][0].dtype)
    
    metric = {}
    loss = {}
    if rpn_score is not None and rpn_regress is not None:
        anchors = tf.tile(tf.expand_dims(anchors, axis = 0), [tf.shape(input)[0], 1, 1])
        rpn_match, rpn_bbox_true, rpn_score, rpn_bbox_pred = tf.keras.layers.Lambda(lambda args: map_fn(rpn_target, *args, dtype = (tf.int32, bbox_true.dtype, rpn_score.dtype, rpn_regress.dtype), batch_size = batch_size, 
                                                                                                        assign = rpn_assign, sampling_count = sampling_count, positive_ratio = rpn_positive_ratio, valid = valid, mean = rpn_mean, std = rpn_std), name = "rpn_target")([bbox_true, rpn_score, rpn_regress, anchors])
        
        rpn_score_accuracy = tf.keras.layers.Lambda(lambda args: score_accuracy(*args, threshold = threshold, missing_value = missing_value), name = "rpn_score_accuracy")([rpn_match, rpn_score])
        rpn_score_loss = tf.keras.layers.Lambda(lambda args: score_loss(*args, focal = focal, missing_value = missing_value), name = "rpn_score_loss")([rpn_match, rpn_score])
        rpn_regress_loss = tf.keras.layers.Lambda(lambda args: regress_loss(*args, sigma = sigma, missing_value = missing_value), name = "rpn_regress_loss")([rpn_match, rpn_bbox_true, rpn_bbox_pred])
        metric["rpn_score_accuracy"] = rpn_score_accuracy
        loss["rpn_score_loss"] = rpn_score_loss
        loss["rpn_regress_loss"] = rpn_regress_loss
    
    if cls_logits is not None and cls_regress is not None and proposals is not None:
        if not isinstance(cls_logits, list):
            cls_logits, cls_regress, mask_regress = [cls_logits], [cls_regress], [mask_regress]
            if not isinstance(proposals, list):
                proposals = [proposals]
        if mask_regress is None:
            mask_regress = [None] * len(cls_logits)
        if isinstance(sampling_tag, dict):
            cls_assign = list(sampling_tag["sampling_assign"])
        else:
            if not isinstance(cls_assign, list):
                cls_assign = [cls_assign]
            if len(cls_assign) == 1:
                cls_assign = list(cls_assign) * len(cls_logits)
        
        #len(proposals) > 1 = faster_rcnn or mask_rcnn 2 = mask_rcnn + interleaved, 3 = cascade_rcnn, 4 = cascade_rcnn + mask + interleaved(hybrid_task_cascade_rcnn)
        #len(cls_logits) > 1 = faster_rcnn or mask_rcnn + @, 3 = cascade_rcnn + @
        interleaved = True if len(proposals) in [2, 4] else False
        if not isinstance(sampling_tag, dict):
            cls_y_true = y_true = tf.keras.layers.Input(shape = (None, None), name = "y_true", dtype = cls_logits[0].dtype)
            cls_bbox_true = bbox_true
            if mask_regress[-1] is not None or semantic_regress is not None:
                cls_mask_true = mask_true = tf.keras.layers.Input(shape = (None, None, None, 1), name = "mask_true", dtype = [l for l in [mask_regress[-1], semantic_regress] if l is not None][0].dtype)
        
        cls_dtype = (y_true.dtype, bbox_true.dtype, cls_logits[0].dtype, cls_regress[0].dtype)
        sampling_func = lambda args, **kwargs: map_fn(sampling_postprocess, *args, dtype = cls_dtype, batch_size = batch_size, **kwargs)
        cls_func = lambda args, **kwargs: map_fn(cls_target, *args, dtype = cls_dtype, batch_size = batch_size, **kwargs)
        if mask_regress[-1] is not None:
            mask_dtype = (y_true.dtype, bbox_true.dtype, mask_true.dtype, cls_logits[0].dtype, cls_regress[0].dtype, mask_regress[-1].dtype)
            sampling_mask_func = lambda args, **kwargs: map_fn(sampling_postprocess, *args, dtype = mask_dtype, batch_size = batch_size, **kwargs)
            cls_mask_func = lambda args, **kwargs: map_fn(cls_target, *args, dtype = mask_dtype, batch_size = batch_size, **kwargs)

        for i, (_cls_logits, _cls_regress, _proposals, _mask_regress) in enumerate(zip(cls_logits, cls_regress, proposals[:len(cls_logits)], mask_regress)):
            if isinstance(sampling_tag, dict):
                cls_y_true, cls_bbox_true, cls_mask_true = sampling_y_true[i], sampling_bbox_true[i], sampling_mask_true[i]
                kwargs = {"interleaved":interleaved, "mean":cls_mean, "std":np.divide(cls_std, i + 1), "clip_ratio":cls_clip_ratio, "method":method}
                if _mask_regress is not None:
                    cls_y_true, cls_bbox_true, cls_mask_true, cls_y_pred, cls_bbox_pred, cls_mask_pred = tf.keras.layers.Lambda(sampling_mask_func, arguments = kwargs, name = "sampling_postprocess{0}".format(i + 1) if 1 < len(proposals) else "sampling_postprocess")([cls_y_true, cls_bbox_true, _cls_logits, _cls_regress, _proposals, cls_mask_true, _mask_regress])
                else:
                    cls_y_true, cls_bbox_true, cls_y_pred, cls_bbox_pred = tf.keras.layers.Lambda(sampling_func, arguments = kwargs, name = "sampling_postprocess{0}".format(i + 1) if 1 < len(proposals) else "sampling_postprocess")([cls_y_true, cls_bbox_true, _cls_logits, _cls_regress, _proposals])
            #elif _cls_logits is not None:
            else:
                kwargs = {"assign":cls_assign[i], "sampling_count":sampling_count, "positive_ratio":cls_positive_ratio, "interleaved":interleaved, "mean":cls_mean, "std":np.divide(cls_std, i + 1), "clip_ratio":cls_clip_ratio, "method":method}
                if _mask_regress is not None:
                    cls_y_true, cls_bbox_true, cls_mask_true, cls_y_pred, cls_bbox_pred, cls_mask_pred = tf.keras.layers.Lambda(cls_mask_func, arguments = kwargs, name = "cls_target{0}".format(i + 1) if 1 < len(proposals) else "cls_target")([cls_y_true, cls_bbox_true, _cls_logits, _cls_regress, _proposals, cls_mask_true, _mask_regress])
                else:
                    cls_y_true, cls_bbox_true, cls_y_pred, cls_bbox_pred = tf.keras.layers.Lambda(cls_func, arguments = kwargs, name = "cls_target{0}".format(i + 1) if 1 < len(proposals) else "cls_target")([cls_y_true, cls_bbox_true, _cls_logits, _cls_regress, _proposals])
            
            if _cls_logits is not None:
                cls_logits_accuracy = tf.keras.layers.Lambda(lambda args: logits_accuracy(*args, missing_value = missing_value), name = "cls_logits_accuracy{0}".format(i + 1) if 2 < len(proposals) else "cls_logits_accuracy")([cls_y_true, cls_y_pred])
                metric["cls_logits_accuracy{0}".format(i + 1) if 2 < len(proposals) else "cls_logits_accuracy"] = cls_logits_accuracy

                cls_logits_loss = tf.keras.layers.Lambda(lambda args: logits_loss(*args, focal = focal, alpha = alpha, gamma = gamma, weight = class_weight, missing_value = missing_value), name = "cls_logits_loss{0}".format(i + 1) if 2 < len(proposals) else "cls_logits_loss")([cls_y_true, cls_y_pred])
                cls_regress_loss = tf.keras.layers.Lambda(lambda args: regress_loss(*args, sigma = sigma, missing_value = missing_value), name = "cls_regress_loss{0}".format(i + 1) if 2 < len(proposals) else "cls_regress_loss")([cls_y_true, cls_bbox_true, cls_bbox_pred])
                loss["cls_logits_loss{0}".format(i + 1) if 2 < len(proposals) else "cls_logits_loss"] = cls_logits_loss * stage_weight[i]
                loss["cls_regress_loss{0}".format(i + 1) if 2 < len(proposals) else "cls_regress_loss"] = cls_regress_loss * stage_weight[i]
            
            if _mask_regress is not None:
                mask_index = i + 1 if mask_regress[0] is not None else i
                cls_mask_loss = tf.keras.layers.Lambda(lambda args: mask_loss(*args, missing_value = missing_value), name = "cls_mask_loss{0}".format(mask_index) if 2 < len(proposals) else "cls_mask_loss")([cls_y_true, cls_mask_true, cls_mask_pred])
                loss["cls_mask_loss{0}".format(mask_index) if 2 < len(proposals) else "cls_mask_loss"] = cls_mask_loss * stage_weight[i]

        if semantic_regress is not None:
            _semantic_loss = tf.keras.layers.Lambda(lambda args: semantic_loss(*args, method = method, weight = class_weight, missing_value = missing_value), name = "semantic_loss")([y_true, mask_true, semantic_regress])
            loss["semantic_loss"] = _semantic_loss * semantic_weight
    
    metric = {k:tf.expand_dims(v, axis = -1) for k, v in metric.items()}
    loss = {k:tf.expand_dims(v, axis = -1) for k, v in loss.items()}

    input = [input] + [l for l in [y_true, bbox_true, mask_true] if l is not None]
    output = FilterDetection(proposal_count = proposal_count, iou_threshold = iou_threshold, score_threshold = score_threshold, soft_nms = soft_nms, ensemble = ensemble, valid = valid, ignore_label = ignore_label, performance_count = performance_count,
                             batch_size = batch_size, mean = cls_mean, std = cls_std, clip_ratio = cls_clip_ratio)(args)
    model = tf.keras.Model(input, list(output))

    for k, v in list(metric.items()) + list(loss.items()):
        model.add_metric(v, name = k, aggregation = "mean")
    
    for k, v in loss.items():
        model.add_loss(v)

    if regularize:
        model.add_loss(lambda: tf.reduce_sum(regularize_loss(model, weight_decay), keepdims = True, name = "regularize_loss"))
    return model