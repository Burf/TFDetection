import tensorflow as tf

from ..util.bbox import bbox2delta
from ..util.overlap import overlap_bbox

def rpn_target(bbox_true, rpn_score, rpn_regress, anchors, sampling_count = 256, positive_ratio = 0.5, positive_threshold = 0.7, negative_threshold = 0.3, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2]):
    """
    bbox_true = [[x1, y1, x2, y2], ...] #(padded_num_true, bbox)
    rpn_score = score for FG/BG #(num_anchors, 1)
    rpn_regress = rpn regress #(num_anchors, delta)
    anchors = [[x1, y1, x2, y2], ...] #(num_anchors, bbox)

    y_true = -1 : negative / 0 : neutral / 1 : positive #(sampling_count, 1)
    box_true = [[x1, y1, x2, y2], ...] #(sampling_count, delta)
    y_pred = -1 : negative / 0 : neutral / 1 : positive #(sampling_count, 1)
    bbox_pred = [[x1, y1, x2, y2], ...] #(sampling_count, delta)
    """
    pred_count = tf.shape(anchors)[0]
    valid_indices = tf.where(tf.reduce_max(tf.cast(0 < bbox_true, tf.int32), axis = -1))
    bbox_true = tf.gather_nd(bbox_true, valid_indices)

    overlaps = overlap_bbox(bbox_true, anchors)
    max_iou = tf.reduce_max(overlaps, axis = -1)
    
    positive_match = tf.where(positive_threshold <= max_iou, 1, 0)
    negative_match = tf.where(max_iou < negative_threshold, -1, 0)
    rpn_match = tf.expand_dims(positive_match + negative_match, axis = -1)
  
    positive_indices = tf.where(rpn_match == 1)[:, 0]
    negative_indices = tf.where(rpn_match == -1)[:, 0]
    
    if sampling_count is not None:
        positive_count = tf.cast(sampling_count * positive_ratio, tf.int32)
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
        positive_count = tf.cast(tf.shape(positive_indices)[0], tf.float32)
        negative_count = tf.cast(1 / positive_ratio * positive_count - positive_count, tf.int32)
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
    else:
        sampling_count = pred_count
    pred_indices = tf.concat([positive_indices, negative_indices], axis = 0)
    
    positive_overlaps = tf.gather(overlaps, positive_indices)
    true_indices = tf.cond(tf.greater(tf.shape(positive_overlaps)[1], 0), true_fn = lambda: tf.argmax(positive_overlaps, axis = -1), false_fn = lambda: tf.cast(tf.constant([]), tf.int64))
    y_true = tf.gather(rpn_match, pred_indices)
    bbox_true = tf.gather(bbox_true, true_indices)
    y_pred = tf.gather(rpn_score, pred_indices)
    bbox_pred = tf.gather(rpn_regress, positive_indices)
    anchors = tf.gather(anchors, positive_indices)
    if tf.keras.backend.int_shape(true_indices)[0] != 0:
        bbox_true = bbox2delta(bbox_true, anchors, mean, std)
    
    negative_count = tf.shape(negative_indices)[0]
    pad_count = tf.maximum(sampling_count - tf.shape(pred_indices)[0], 0)
    y_true = tf.pad(y_true, [[0, pad_count], [0, 0]])
    bbox_true = tf.pad(bbox_true, [[0, negative_count + pad_count], [0, 0]])
    y_pred = tf.pad(y_pred, [[0, pad_count], [0, 0]])
    bbox_pred = tf.pad(bbox_pred, [[0, negative_count + pad_count], [0, 0]])
    return y_true, bbox_true, y_pred, bbox_pred

def sampling_target(y_true, bbox_true, proposal, mask_true = None, sampling_count = 256, positive_ratio = 0.25, positive_threshold = 0.5, negative_threshold = 0.5):
    """
    y_true = label #(padded_num_true, 1 or num_class)
    bbox_true = [[x1, y1, x2, y2], ...] #(padded_num_true, bbox)
    mask_true = mask #(padded_num_true, h, w)
    proposal = [[x1, y1, x2, y2], ...] #(num_proposals, bbox)

    y_true = targeted label #(sampling_count, 1 or num_class) 
    bbox_true = [[x1, y1, x2, y2], ...] #(sampling_count, bbox)
    mask_true = targeted mask true #(sampling_count, h, w)
    proposal = [[x1, y1, x2, y2], ...] #(sampling_count, bbox)
    """
    pred_count = tf.shape(proposal)[0]
    valid_true_indices = tf.where(tf.reduce_max(tf.cast(0 < bbox_true, tf.int32), axis = -1))
    y_true = tf.gather_nd(y_true, valid_true_indices)
    bbox_true = tf.gather_nd(bbox_true, valid_true_indices)
    valid_pred_indices = tf.where(tf.reduce_max(tf.cast(0 < proposal, tf.int32), axis = -1))
    proposal = tf.gather_nd(proposal, valid_pred_indices)
    if mask_true is not None:
        mask_true = tf.gather_nd(mask_true, valid_true_indices)

    overlaps = overlap_bbox(bbox_true, proposal)
    max_iou = tf.reduce_max(overlaps, axis = -1)

    positive_indices = tf.where(positive_threshold <= max_iou)[:, 0]
    negative_indices = tf.where(max_iou < negative_threshold)[:, 0]
    
    if sampling_count is not None:
        positive_count = tf.cast(sampling_count * positive_ratio, tf.int32)
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
        positive_count = tf.cast(tf.shape(positive_indices)[0], tf.float32)
        negative_count = tf.cast(1 / positive_ratio * positive_count - positive_count, tf.int32)
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
    else:
        sampling_count = pred_count
    pred_indices = tf.concat([positive_indices, negative_indices], axis = 0)
    
    positive_overlaps = tf.gather(overlaps, positive_indices)
    true_indices = tf.cond(tf.greater(tf.shape(positive_overlaps)[1], 0), true_fn = lambda: tf.argmax(positive_overlaps, axis = -1), false_fn = lambda: tf.cast(tf.constant([]), tf.int64))
    y_true = tf.gather(y_true, true_indices)
    bbox_true = tf.gather(bbox_true, true_indices)
    proposal = tf.gather(proposal, pred_indices)
    if mask_true is not None:
        mask_true = tf.gather(mask_true, true_indices)
    
    n_class = tf.shape(y_true)[-1]
    negative_count = tf.shape(negative_indices)[0]
    pad_count = tf.maximum(sampling_count - tf.shape(pred_indices)[0], 0)
    y_true = tf.cond(tf.equal(n_class, 1), true_fn = lambda: tf.pad(y_true, [[0, negative_count + pad_count], [0, 0]]), false_fn = lambda: tf.concat([y_true, tf.cast(tf.pad(tf.ones([negative_count + pad_count, 1]), [[0, 0], [0, n_class - 1]]), y_true.dtype)], axis = 0))
    bbox_true = tf.pad(bbox_true, [[0, negative_count + pad_count], [0, 0]])
    proposal = tf.pad(proposal, [[0, pad_count], [0, 0]])
    result = y_true, bbox_true, proposal
    if mask_true is not None:
        mask_true = tf.pad(mask_true, [[0, negative_count + pad_count], [0, 0], [0, 0]])
        result = y_true, bbox_true, mask_true, proposal
    return result

def sampling_postprocess(y_true, bbox_true, cls_logits, cls_regress, proposal, mask_true = None, mask_regress = None, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], method = "bilinear"):
    if mask_true is not None:
        if tf.keras.backend.ndim(mask_true) == 3:
            mask_true = tf.expand_dims(mask_true, axis = -1)

    sampling_count = tf.shape(proposal)[0]
    positive_indices = tf.where(tf.reduce_max(tf.cast(0 < bbox_true, tf.int32), axis = -1))
    pred_indices = tf.where(tf.reduce_max(tf.cast(0 < proposal, tf.int32), axis = -1))
    y_true = tf.gather_nd(y_true, positive_indices)
    bbox_true = tf.gather_nd(bbox_true, positive_indices)
    y_pred = tf.gather_nd(cls_logits, pred_indices)
    bbox_pred = tf.gather_nd(cls_regress, positive_indices)
    proposal = tf.gather_nd(proposal, positive_indices)
    if mask_true is not None and mask_regress is not None:
        mask_true = tf.gather_nd(mask_true, positive_indices)
        mask_pred = tf.gather_nd(mask_regress, positive_indices)

    positive_count = tf.shape(y_true)[0]
    n_class = tf.shape(y_true)[-1]
    if tf.keras.backend.int_shape(positive_indices)[0] != 0:
        bbox_true = bbox2delta(bbox_true, proposal, mean, std)
        label = tf.cond(tf.equal(n_class, 1), true_fn = lambda: y_true, false_fn = lambda: tf.expand_dims(tf.cast(tf.argmax(y_true, axis = -1), y_true.dtype), axis = -1))    
        indices = tf.stack([tf.range(tf.shape(label)[0]), tf.cast(label[:, 0], tf.int32)], axis = -1)
        bbox_pred = tf.gather_nd(bbox_pred, indices)
        if mask_true is not None and mask_regress is not None:
            x1, y1, x2, y2 = tf.split(proposal, 4, axis = -1)
            mask_bbox = tf.concat([y1, x1, y2, x2], axis = -1)
            mask_shape = tf.shape(mask_pred)
            mask_true = tf.image.crop_and_resize(image = tf.cast(mask_true, mask_pred.dtype), boxes = mask_bbox, box_indices = tf.range(0, positive_count), crop_size = mask_shape[1:3], method = method)
            mask_true = mask_true[..., 0]
            mask_true = tf.clip_by_value(tf.round(mask_true), 0., 1.)
            mask_pred = tf.transpose(mask_pred, [0, 3, 1, 2])
            mask_pred = tf.gather_nd(mask_pred, indices)
    else:
        bbox_pred = bbox_pred[:, 0]
        if mask_true is not None and mask_regress is not None:
            mask_pred = mask_pred[..., 0]
            mask_true = tf.zeros_like(mask_pred, dtype = mask_pred.dtype)
    
    pred_count = tf.shape(pred_indices)[0]
    pad_count = sampling_count - pred_count
    negative_count = tf.maximum(pred_count - positive_count , 0)
    y_true = tf.cond(tf.equal(n_class, 1), true_fn = lambda: tf.pad(y_true, [[0, negative_count + pad_count], [0, 0]]), false_fn = lambda: tf.concat([y_true, tf.cast(tf.pad(tf.ones([negative_count + pad_count, 1]), [[0, 0], [0, n_class - 1]]), y_true.dtype)], axis = 0))
    y_pred = tf.pad(y_pred, [[0, pad_count], [0, 0]])
    bbox_true = tf.pad(bbox_true, [[0, negative_count + pad_count], [0, 0]])
    bbox_pred = tf.pad(bbox_pred, [[0, negative_count + pad_count], [0, 0]])
    result = y_true, bbox_true, y_pred, bbox_pred
    if mask_true is not None and mask_regress is not None:
        mask_true = tf.pad(mask_true, [[0, negative_count + pad_count], [0, 0], [0, 0]])
        mask_pred = tf.pad(mask_pred, [[0, negative_count + pad_count], [0, 0], [0, 0]])
        result = y_true, bbox_true, mask_true, y_pred, bbox_pred, mask_pred
    return result

def cls_target(y_true, bbox_true, cls_logit, cls_regress, proposal, mask_true = None, mask_regress = None, sampling_count = 256, positive_ratio = 0.25, positive_threshold = 0.5, negative_threshold = 0.5, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], method = "bilinear"):
    """
    y_true = label #(padded_num_true, 1 or num_class)
    bbox_true = [[x1, y1, x2, y2], ...] #(padded_num_true, bbox)
    mask_true = mask #(padded_num_true, h, w)
    cls_logit = classifier logit #(num_proposals, num_class)
    cls_regress = classifier regress #(num_proposals, num_class, delta)
    mask_regress = mask regress #(num_proposals, h, w, num_class)
    proposal = [[x1, y1, x2, y2], ...] #(num_proposals, bbox)

    y_true = targeted label #(sampling_count, 1 or num_class) 
    bbox_true = [[x1, y1, x2, y2], ...] #(sampling_count, delta)
    mask_true = targeted mask true #(sampling_count, h, w)
    y_pred = targeted logit #(sampling_count, num_class)
    bbox_pred = [[x1, y1, x2, y2], ...] #(sampling_count, delta)
    mask_pred = targeted mask regress #(sampling_count, h, w)
    """
    if mask_true is not None and mask_regress is not None:
        if tf.keras.backend.ndim(mask_true) == 3:
            mask_true = tf.expand_dims(mask_true, axis = -1)
    
    pred_count = tf.shape(proposal)[0]
    valid_true_indices = tf.where(tf.reduce_max(tf.cast(0 < bbox_true, tf.int32), axis = -1))
    y_true = tf.gather_nd(y_true, valid_true_indices)
    bbox_true = tf.gather_nd(bbox_true, valid_true_indices)
    valid_pred_indices = tf.where(tf.reduce_max(tf.cast(0 < proposal, tf.int32), axis = -1))
    cls_logit = tf.gather_nd(cls_logit, valid_pred_indices)
    cls_regress = tf.gather_nd(cls_regress, valid_pred_indices)
    proposal = tf.gather_nd(proposal, valid_pred_indices)
    if mask_true is not None and mask_regress is not None:
        mask_true = tf.gather_nd(mask_true, valid_true_indices)
        mask_regress = tf.gather_nd(mask_regress, valid_pred_indices)

    overlaps = overlap_bbox(bbox_true, proposal)
    max_iou = tf.reduce_max(overlaps, axis = -1)

    positive_indices = tf.where(positive_threshold <= max_iou)[:, 0]
    negative_indices = tf.where(max_iou < negative_threshold)[:, 0]
    
    if sampling_count is not None:
        positive_count = tf.cast(sampling_count * positive_ratio, tf.int32)
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
        positive_count = tf.cast(tf.shape(positive_indices)[0], tf.float32)
        negative_count = tf.cast(1 / positive_ratio * positive_count - positive_count, tf.int32)
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
    else:
        sampling_count = pred_count
        positive_count = tf.shape(positive_indices)[0]
    pred_indices = tf.concat([positive_indices, negative_indices], axis = 0)
    
    positive_overlaps = tf.gather(overlaps, positive_indices)
    true_indices = tf.cond(tf.greater(tf.shape(positive_overlaps)[1], 0), true_fn = lambda: tf.argmax(positive_overlaps, axis = -1), false_fn = lambda: tf.cast(tf.constant([]), tf.int64))
    y_true = tf.gather(y_true, true_indices)
    bbox_true = tf.gather(bbox_true, true_indices)
    y_pred = tf.gather(cls_logit, pred_indices)
    bbox_pred = tf.gather(cls_regress, positive_indices)
    proposal = tf.gather(proposal, positive_indices)
    if mask_true is not None and mask_regress is not None:
        mask_true = tf.gather(mask_true, true_indices)
        mask_pred = tf.gather(mask_regress, positive_indices)

    n_class = tf.shape(y_true)[-1]
    if tf.keras.backend.int_shape(true_indices)[0] != 0:
        bbox_true = bbox2delta(bbox_true, proposal, mean, std)
        label = tf.cond(tf.equal(n_class, 1), true_fn = lambda: y_true, false_fn = lambda: tf.expand_dims(tf.cast(tf.argmax(y_true, axis = -1), y_true.dtype), axis = -1))    
        indices = tf.stack([tf.range(tf.shape(label)[0]), tf.cast(label[:, 0], tf.int32)], axis = -1)
        bbox_pred = tf.gather_nd(bbox_pred, indices)
        if mask_true is not None and mask_regress is not None:
            x1, y1, x2, y2 = tf.split(proposal, 4, axis = -1)
            mask_bbox = tf.concat([y1, x1, y2, x2], axis = -1)
            mask_shape = tf.shape(mask_pred)
            mask_true = tf.image.crop_and_resize(image = tf.cast(mask_true, mask_pred.dtype), boxes = mask_bbox, box_indices = tf.range(0, tf.cast(positive_count, tf.int32)), crop_size = mask_shape[1:3], method = method)
            mask_true = mask_true[..., 0]
            mask_true = tf.clip_by_value(tf.round(mask_true), 0., 1.)
            mask_pred = tf.transpose(mask_pred, [0, 3, 1, 2])
            mask_pred = tf.gather_nd(mask_pred, indices)
    else:
        bbox_pred = bbox_pred[:, 0]
        if mask_true is not None and mask_regress is not None:
            mask_pred = mask_pred[..., 0]
            mask_true = tf.zeros_like(mask_pred, dtype = mask_pred.dtype)
    
    negative_count = tf.shape(negative_indices)[0]
    pad_count = tf.maximum(sampling_count - tf.shape(pred_indices)[0], 0)
    y_true = tf.cond(tf.equal(n_class, 1), true_fn = lambda: tf.pad(y_true, [[0, negative_count + pad_count], [0, 0]]), false_fn = lambda: tf.concat([y_true, tf.cast(tf.pad(tf.ones([negative_count + pad_count, 1]), [[0, 0], [0, n_class - 1]]), y_true.dtype)], axis = 0))
    bbox_true = tf.pad(bbox_true, [[0, negative_count + pad_count], [0, 0]])
    y_pred = tf.pad(y_pred, [[0, pad_count], [0, 0]])
    bbox_pred = tf.pad(bbox_pred, [[0, negative_count + pad_count], [0, 0]])
    result = y_true, bbox_true, y_pred, bbox_pred
    if mask_true is not None and mask_regress is not None:
        mask_true = tf.pad(mask_true, [[0, negative_count + pad_count], [0, 0], [0, 0]])
        mask_pred = tf.pad(mask_pred, [[0, negative_count + pad_count], [0, 0], [0, 0]])
        result = y_true, bbox_true, mask_true, y_pred, bbox_pred, mask_pred
    return result

def mask_target(y_true, bbox_true, mask_true, mask_regress, proposal, sampling_count = 256, positive_ratio = 0.25, positive_threshold = 0.5, negative_threshold = 0.5, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], method = "bilinear"):
    """
    y_true = label #(padded_num_true, 1 or num_class)
    bbox_true = [[x1, y1, x2, y2], ...] #(padded_num_true, bbox)
    mask_true = mask #(padded_num_true, h, w)
    mask_regress = mask regress #(num_proposals, h, w, num_class)
    proposal = [[x1, y1, x2, y2], ...] #(num_proposals, bbox)

    mask_true = targeted mask true #(sampling_count, h, w)
    mask_pred = targeted mask regress #(sampling_count, h, w)
    """
    if tf.keras.backend.ndim(mask_true) == 3:
        mask_true = tf.expand_dims(mask_true, axis = -1)
    
    pred_count = tf.shape(proposal)[0]
    valid_true_indices = tf.where(tf.reduce_max(tf.cast(0 < bbox_true, tf.int32), axis = -1))
    y_true = tf.gather_nd(y_true, valid_true_indices)
    bbox_true = tf.gather_nd(bbox_true, valid_true_indices)
    valid_pred_indices = tf.where(tf.reduce_max(tf.cast(0 < proposal, tf.int32), axis = -1))
    proposal = tf.gather_nd(proposal, valid_pred_indices)
    mask_true = tf.gather_nd(mask_true, valid_true_indices)
    mask_regress = tf.gather_nd(mask_regress, valid_pred_indices)

    overlaps = overlap_bbox(bbox_true, proposal)
    max_iou = tf.reduce_max(overlaps, axis = -1)

    positive_indices = tf.where(positive_threshold <= max_iou)[:, 0]
    negative_indices = tf.where(max_iou < negative_threshold)[:, 0]
    
    if sampling_count is not None:
        positive_count = tf.cast(sampling_count * positive_ratio, tf.int32)
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
        positive_count = tf.cast(tf.shape(positive_indices)[0], tf.float32)
        negative_count = tf.cast(1 / positive_ratio * positive_count - positive_count, tf.int32)
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
    else:
        sampling_count = pred_count
        positive_count = tf.shape(positive_indices)[0]
    pred_indices = tf.concat([positive_indices, negative_indices], axis = 0)
    
    positive_overlaps = tf.gather(overlaps, positive_indices)
    true_indices = tf.cond(tf.greater(tf.shape(positive_overlaps)[1], 0), true_fn = lambda: tf.argmax(positive_overlaps, axis = -1), false_fn = lambda: tf.cast(tf.constant([]), tf.int64))
    y_true = tf.gather(y_true, true_indices)
    proposal = tf.gather(proposal, positive_indices)
    mask_true = tf.gather(mask_true, true_indices)
    mask_pred = tf.gather(mask_regress, positive_indices)

    n_class = tf.shape(y_true)[-1]
    if tf.keras.backend.int_shape(true_indices)[0] != 0:
        label = tf.cond(tf.equal(n_class, 1), true_fn = lambda: y_true, false_fn = lambda: tf.expand_dims(tf.cast(tf.argmax(y_true, axis = -1), y_true.dtype), axis = -1))    
        indices = tf.stack([tf.range(tf.shape(label)[0]), tf.cast(label[:, 0], tf.int32)], axis = -1)
        if mask_true is not None and mask_regress is not None:
            x1, y1, x2, y2 = tf.split(proposal, 4, axis = -1)
            mask_bbox = tf.concat([y1, x1, y2, x2], axis = -1)
            mask_shape = tf.shape(mask_pred)
            mask_true = tf.image.crop_and_resize(image = tf.cast(mask_true, mask_pred.dtype), boxes = mask_bbox, box_indices = tf.range(0, tf.cast(positive_count, tf.int32)), crop_size = mask_shape[1:3], method = method)
            mask_true = mask_true[..., 0]
            mask_true = tf.clip_by_value(tf.round(mask_true), 0., 1.)
            mask_pred = tf.transpose(mask_pred, [0, 3, 1, 2])
            mask_pred = tf.gather_nd(mask_pred, indices)
    else:
        mask_pred = mask_pred[..., 0]
        mask_true = tf.zeros_like(mask_pred, dtype = mask_pred.dtype)
    
    negative_count = tf.shape(negative_indices)[0]
    pad_count = tf.maximum(sampling_count - tf.shape(pred_indices)[0], 0)
    mask_true = tf.pad(mask_true, [[0, negative_count + pad_count], [0, 0], [0, 0]])
    mask_pred = tf.pad(mask_pred, [[0, negative_count + pad_count], [0, 0], [0, 0]])
    return mask_true, mask_pred