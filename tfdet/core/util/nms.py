import tensorflow as tf

from ..bbox.coder import delta2bbox

def pad_nms(proposal, score, proposal_count, iou_threshold, score_threshold = float('-inf'), soft_nms = False):
    soft_nms_sigma = soft_nms
    if not isinstance(soft_nms, float):
        soft_nms_sigma = 0.5 if soft_nms else 0.
    indices = tf.image.non_max_suppression_with_scores(proposal, score, max_output_size = tf.minimum(proposal_count, tf.shape(proposal)[0]), iou_threshold = iou_threshold, score_threshold = score_threshold, soft_nms_sigma = soft_nms_sigma)[0]
    proposal = tf.gather(proposal, indices)
    pad_size = proposal_count - tf.shape(proposal)[0]
    proposal = tf.pad(proposal, [[0, pad_size], [0, 0]])
    return proposal

def multiclass_nms(y_pred, bbox_pred, anchors, mask_pred = None, proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05, soft_nms = False, performance_count = 5000,
                   coder_func = delta2bbox, **kwargs):
    """
    y_pred = logit #(n_anchor, n_class)
    bbox_pred = delta #(n_anchor, 4) or n_class delta #(n_anchor, n_clss, 4)
    anchors = anchors #(n_anchor, 4) or points #(n_anchor, 2)
    mask_pred = mask #(n_anchor, H, W, 1 or n_class)

    y_pred = logit #(proposal_count, n_class)
    bbox_pred = normalized proposal [[x1, y1, x2, y2], ...] #(proposal_count, 4)
    """
    n_class = tf.keras.backend.int_shape(y_pred)[-1]
    score_threshold = [score_threshold] * n_class if isinstance(score_threshold, float) else score_threshold
    soft_nms_sigma = soft_nms
    if not isinstance(soft_nms, float):
        soft_nms_sigma = 0.5 if soft_nms else 0.
    
    #remove background
    indices = tf.where(0 < tf.argmax(y_pred, axis = -1))
    y_pred = tf.gather_nd(y_pred, indices)
    bbox_pred = tf.gather_nd(bbox_pred, indices)
    anchors = tf.gather_nd(anchors, indices)
    if mask_pred is not None:
        mask_pred = tf.gather_nd(mask_pred, indices)
    
    #reduce by performance_count
    performance_count = tf.minimum(performance_count, tf.shape(y_pred)[0])
    top_indices = tf.nn.top_k(tf.reduce_max(y_pred, axis = -1), performance_count, sorted = True).indices
    y_pred = tf.gather(y_pred, top_indices)
    bbox_pred = tf.gather(bbox_pred, top_indices)
    anchors = tf.gather(anchors, top_indices)
    if mask_pred is not None:
        mask_pred = tf.gather(mask_pred, top_indices)
          
    bbox_flag = (tf.keras.backend.ndim(bbox_pred) == 3)
    mask_flag = (mask_pred is not None and tf.keras.backend.int_shape(mask_pred)[-1] != 1)
    if bbox_flag or mask_flag:
        label_indices = tf.stack([tf.range(tf.shape(bbox_pred)[0]), tf.argmax(y_pred, axis = -1, output_type = tf.int32)], axis = -1)
        if bbox_flag:
            bbox_pred = tf.gather_nd(bbox_pred, label_indices)
        if mask_flag:
            mask_pred = tf.transpose(mask_pred, [0, 3, 1, 2])
            mask_pred = tf.gather_nd(mask_pred, label_indices)
            mask_pred = tf.expand_dims(mask_pred, axis = -1)
        
    if callable(coder_func):
        bbox_pred = coder_func(anchors, bbox_pred, **kwargs)
    bbox_pred = tf.clip_by_value(bbox_pred, 0, 1)
    x1, y1, x2, y2 = tf.split(bbox_pred, 4, axis = -1)
    bbox = tf.concat([y1, x1, y2, x2], axis = -1)
    
    indices = []
    for cls in range(1, n_class):
        threshold = score_threshold[cls]
        score = y_pred[..., cls]
        cls_indices = tf.image.non_max_suppression_with_scores(bbox, score, max_output_size = proposal_count, iou_threshold = iou_threshold, score_threshold = threshold, soft_nms_sigma = soft_nms_sigma)[0]
        cls_indices = tf.stack([cls_indices, tf.cast(tf.fill(tf.shape(cls_indices), cls), tf.int32)], axis = -1)
        #pad_size = proposal_count - tf.shape(cls_indices)[0]
        #cls_indices = tf.pad(cls_indices, [[0, pad_size], [0, 0]], constant_values = -1)
        #cls_indices = tf.reshape(cls_indices, [proposal_count, 2])
        indices.append(cls_indices)
    indices = tf.concat(indices, axis = 0)
    #remove_indices = tf.where(-1 < indices[..., 0])
    #indices = tf.gather_nd(indices, remove_indices)
    
    score = tf.gather_nd(y_pred, indices)
    top_indices = tf.nn.top_k(score, tf.minimum(proposal_count, tf.shape(score)[0]), sorted = True).indices
    indices = tf.gather(indices[:, 0], top_indices)
    y_pred = tf.gather(y_pred, indices)
    bbox_pred = tf.gather(bbox_pred, indices)
    if mask_pred is not None:
        h, w = tf.keras.backend.int_shape(mask_pred)[-3:-1]
        mask_pred = tf.gather(mask_pred, indices)
    
    pad_count = tf.maximum(proposal_count - tf.shape(bbox_pred)[0], 0)
    y_pred = tf.pad(y_pred, [[0, pad_count], [0, 0]])
    bbox_pred = tf.pad(bbox_pred, [[0, pad_count], [0, 0]])
    
    y_pred = tf.reshape(y_pred, [proposal_count, n_class])
    bbox_pred = tf.reshape(bbox_pred, [proposal_count, 4])
    result = y_pred, bbox_pred
    if mask_pred is not None:
        mask_pred = tf.pad(mask_pred, [[0, pad_count], [0, 0], [0, 0], [0, 0]])
        mask_pred = tf.reshape(mask_pred, [proposal_count, h, w, 1])
        result = y_pred, bbox_pred, mask_pred
    return result