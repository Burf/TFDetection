import tensorflow as tf

from tfdet.core.target import rpn_target, sampling_postprocess, cls_target, mask_target
from tfdet.core.util.loss import regularize_loss
from tfdet.core.util.tf import map_fn
from ..loss.rcnn import score_accuracy, score_loss, logits_accuracy, logits_loss, regress_loss, mask_loss, semantic_loss

def train_model(input, rpn_score = None, rpn_regress = None, cls_logits = None, cls_regress = None, proposals = None, anchors = None, mask_regress = None, semantic_regress = None,
                sampling_tag = None, sampling_count = 256,
                rpn_positive_ratio = 0.5, rpn_positive_threshold = 0.7, rpn_negative_threshold = 0.3, 
                cls_positive_ratio = 0.25, cls_positive_threshold = 0.5, cls_negative_threshold = 0.5,
                batch_size = 1, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], method = "bilinear", regularize = True, weight_decay = 1e-4, focal = True, alpha = 1., gamma = 2., sigma = 1, class_weight = None, stage_weight = [1.0, 0.5, 0.25], semantic_weight = 0.2, threshold = 0.5, missing_value = 0.):
    """
    y_true > #(batch_size, padded_num_true, 1 or n_class)
    bbox_true > #(batch_size, padded_num_true, 4)
    mask_true > #(batch_size, padded_num_true, h, w)
    """
    if isinstance(mask_regress, dict):
        sampling_tag = mask_regress
        mask_regress = None
    elif isinstance(semantic_regress, dict):
        sampling_tag = semantic_regress
        semantic_regress = None
        
    y_true = bbox_true = mask_true = sampling_y_true = sampling_bbox_true = sampling_mask_true = None
    if isinstance(sampling_tag, dict):
        sampling_count, cls_positive_ratio, cls_positive_threshold, cls_negative_threshold, y_true, bbox_true, mask_true, sampling_y_true, sampling_bbox_true, sampling_mask_true = [sampling_tag[key] for key in ["sampling_count", "positive_ratio", "positive_threshold", "negative_threshold", 
                                                                                                                                                                                                                   "y_true", "bbox_true", "mask_true", "sampling_y_true", "sampling_bbox_true", "sampling_mask_true"]]
    else:
        bbox_true = tf.keras.layers.Input(shape = (None, 4), name = "bbox_true", dtype = [l[0] if isinstance(l, list) else l for l in [l for l in [rpn_regress, cls_regress] if l is not None]][0].dtype)
    
    metric = {}
    loss = {}
    if rpn_score is not None and rpn_regress is not None:
        anchors = tf.tile(tf.expand_dims(anchors, axis = 0), [tf.shape(input)[0], 1, 1])
        rpn_match, rpn_bbox_true, rpn_score, rpn_bbox_pred = tf.keras.layers.Lambda(lambda args: map_fn(rpn_target, *args, dtype = (tf.int32, bbox_true.dtype, rpn_score.dtype, rpn_regress.dtype), batch_size = batch_size, 
                                                                                                        sampling_count = sampling_count, positive_ratio = rpn_positive_ratio, positive_threshold = rpn_positive_threshold, negative_threshold = rpn_negative_threshold, mean = mean, std = std), name = "rpn_target")([bbox_true, rpn_score, rpn_regress, anchors])
        
        rpn_score_accuracy = tf.keras.layers.Lambda(lambda args: score_accuracy(*args, threshold = threshold, missing_value = missing_value), name = "rpn_score_accuracy")([rpn_match, rpn_score])
        rpn_score_loss = tf.keras.layers.Lambda(lambda args: score_loss(*args, missing_value = missing_value), name = "rpn_score_loss")([rpn_match, rpn_score])
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
            mask_regress = [mask_regress] * len(cls_logits)
        
        #len(proposals) > 1 = faster_rcnn or mask_rcnn 2 = mask_rcnn + mask_info_flow, 3 = cascade_rcnn, 4 = cascade_rcnn + mask + mask_info_flow(hybrid_task_cascade_rcnn)
        #len(cls_logits) > 1 = faster_rcnn or mask_rcnn + @, 3 = cascade_rcnn + @
        cls_positive_threshold = [min(cls_positive_threshold + 0.1 * index, 1.) for index in range(len(cls_logits))] #[cls_positive_threshold, cls_positive_threshold + 0.1, cls_positive_threshold + 0.2]
        cls_negative_threshold = [min(cls_negative_threshold + 0.1 * index, 1.) for index in range(len(cls_logits))] #[cls_negative_threshold, cls_negative_threshold + 0.1, cls_negative_threshold + 0.2]
        stage_weight = [stage_weight] * len(cls_logits) if isinstance(stage_weight, float) else stage_weight
        if len(proposals) in [2, 4]:
            cls_logits, cls_regress, mask_regress = cls_logits + [None], cls_regress + [None], [None] + mask_regress

        if isinstance(sampling_tag, dict):
            cls_y_true, cls_bbox_true, cls_mask_true = sampling_y_true, sampling_bbox_true, sampling_mask_true
        else:
            cls_y_true = y_true = tf.keras.layers.Input(shape = (None, None), name = "y_true", dtype = cls_logits[0].dtype)
            cls_bbox_true = bbox_true
            if mask_regress[-1] is not None or semantic_regress is not None:
                cls_mask_true = mask_true = tf.keras.layers.Input(shape = (None, None, None), name = "mask_true", dtype = mask_regress[-1].dtype)

        sampling_func = lambda args, **kwargs: map_fn(sampling_postprocess, *args, dtype = (cls_y_true.dtype, cls_bbox_true.dtype, cls_logits[0].dtype, cls_regress[0].dtype), batch_size = batch_size, **kwargs)
        sampling_mask_func = lambda args, **kwargs: map_fn(sampling_postprocess, *args, dtype = (cls_y_true.dtype, cls_bbox_true.dtype, cls_mask_true.dtype, cls_logits[0].dtype, cls_regress[0].dtype, mask_regress[-1].dtype), batch_size = batch_size, **kwargs)
        cls_func = lambda args, **kwargs: map_fn(cls_target, *args, dtype = (cls_y_true.dtype, cls_bbox_true.dtype, cls_logits[0].dtype, cls_regress[0].dtype), batch_size = batch_size, **kwargs)
        cls_mask_func = lambda args, **kwargs: map_fn(cls_target, *args, dtype = (cls_y_true.dtype, cls_bbox_true.dtype, cls_mask_true.dtype, cls_logits[0].dtype, cls_regress[0].dtype, mask_regress[-1].dtype), batch_size = batch_size, **kwargs)
        mask_func = lambda args, **kwargs: map_fn(mask_target, *args, dtype = (cls_mask_true.dtype, mask_regress[-1].dtype), batch_size = batch_size, **kwargs)

        for i, (_cls_logits, _cls_regress, _proposals, _mask_regress) in enumerate(zip(cls_logits, cls_regress, proposals, mask_regress)):
            if i == 0 and isinstance(sampling_tag, dict):
                kwargs = {"mean":mean, "std":std, "method":method}
                if _mask_regress is not None:
                    cls_y_true, cls_bbox_true, cls_mask_true, cls_y_pred, cls_bbox_pred, cls_mask_pred = tf.keras.layers.Lambda(sampling_mask_func, arguments = kwargs, name = "cls_target{0}".format(i + 1) if 1 < len(proposals) else "cls_target")([cls_y_true, cls_bbox_true, _cls_logits, _cls_regress, _proposals, cls_mask_true, _mask_regress])
                else:
                    cls_y_true, cls_bbox_true, cls_y_pred, cls_bbox_pred = tf.keras.layers.Lambda(sampling_func, arguments = kwargs, name = "cls_target{0}".format(i + 1) if 1 < len(proposals) else "cls_target")([cls_y_true, cls_bbox_true, _cls_logits, _cls_regress, _proposals])
            elif _cls_logits is not None:
                kwargs = {"sampling_count":sampling_count, "positive_ratio":cls_positive_ratio, "positive_threshold":cls_positive_threshold[i], "negative_threshold":cls_negative_threshold[i], "mean":mean, "std":std, "method":method}
                if _mask_regress is not None:
                    cls_y_true, cls_bbox_true, cls_mask_true, cls_y_pred, cls_bbox_pred, cls_mask_pred = tf.keras.layers.Lambda(cls_mask_func, arguments = kwargs, name = "cls_target{0}".format(i + 1) if 1 < len(proposals) else "cls_target")([cls_y_true, cls_bbox_true, _cls_logits, _cls_regress, _proposals, cls_mask_true, _mask_regress])
                else:
                    cls_y_true, cls_bbox_true, cls_y_pred, cls_bbox_pred = tf.keras.layers.Lambda(cls_func, arguments = kwargs, name = "cls_target{0}".format(i + 1) if 1 < len(proposals) else "cls_target")([cls_y_true, cls_bbox_true, _cls_logits, _cls_regress, _proposals])
            else:
                kwargs = {"sampling_count":sampling_count, "positive_ratio":cls_positive_ratio, "positive_threshold":cls_positive_threshold[-1], "negative_threshold":cls_negative_threshold[-1], "mean":mean, "std":std, "method":method}
                cls_mask_true, cls_mask_pred = tf.keras.layers.Lambda(mask_func, arguments = kwargs, name = "cls_target{0}".format(i + 1) if 1 < len(proposals) else "cls_target")([cls_y_true, cls_bbox_true, cls_mask_true, _mask_regress, _proposals])

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
                loss["cls_mask_loss{0}".format(mask_index) if 2 < len(proposals) else "cls_mask_loss"] = cls_mask_loss

        if semantic_regress is not None:
            _semantic_loss = tf.keras.layers.Lambda(lambda args: semantic_loss(*args), name = "semantic_loss")([mask_true, semantic_regress])
            loss["semantic_loss"] = _semantic_loss * semantic_weight
    
    metric = {k:tf.expand_dims(v, axis = -1) for k, v in metric.items()}
    loss = {k:tf.expand_dims(v, axis = -1) for k, v in loss.items()}

    input = [input] + [l for l in [y_true, bbox_true, mask_true] if l is not None]
    model = tf.keras.Model(input, tf.reduce_sum(list(loss.values()), keepdims = True, name = "loss"))

    for k, v in list(metric.items()) + list(loss.items()):
        model.add_metric(v, name = k, aggregation = "mean")
    
    for k, v in loss.items():
        model.add_loss(v)

    if regularize:
        reg_loss = regularize_loss(model, weight_decay)
        model.add_loss(lambda: tf.reduce_sum(reg_loss, keepdims = True, name = "regularize_loss"))
    return model