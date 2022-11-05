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

def multiclass_nms(y_pred, bbox_pred, anchors = None, mask_pred = None, proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05, soft_nms = False, 
                   ignore_label = 0, performance_count = 5000, coder_func = delta2bbox, **kwargs):
    """
    y_pred = logit #(n_anchor, n_class)
    bbox_pred = delta #(n_anchor, 4) or n_class delta #(n_anchor, n_clss, 4)
    anchors = anchors #(n_anchor, 4) or points #(n_anchor, 2)
    mask_pred = mask #(n_anchor, H, W, 1 or n_class)

    y_pred = logit #(proposal_count, n_class)
    bbox_pred = normalized proposal [[x1, y1, x2, y2], ...] #(proposal_count, 4)
    mask_pred = mask #(proposal_count, H, W, 1)
    """
    n_class = tf.keras.backend.int_shape(y_pred)[-1]
    score_threshold = [score_threshold] * n_class if isinstance(score_threshold, float) else score_threshold
    soft_nms_sigma = soft_nms
    if not isinstance(soft_nms, float):
        soft_nms_sigma = 0.5 if soft_nms else 0.
    ignore_label = [ignore_label] if isinstance(ignore_label, int) else ignore_label
    keep_label = [i for i in range(n_class) if ignore_label is None or i not in ignore_label]
    
    #filtered by label
    if ignore_label is not None:
        label = tf.argmax(y_pred, axis = -1)
        flag = tf.zeros_like(label, dtype = tf.bool)
        for cls in ignore_label:
            flag = tf.logical_or(flag, label == cls)
        indices = tf.where(tf.logical_not(flag))[:, 0]
        
        y_pred = tf.gather(y_pred, indices)
        bbox_pred = tf.gather(bbox_pred, indices)
        if anchors is not None:
            anchors = tf.gather(anchors, indices)
        if mask_pred is not None:
            mask_pred = tf.gather(mask_pred, indices)
    
    #reduce by performance_count
    if isinstance(performance_count, int) and 0 < performance_count:
        top_indices = tf.nn.top_k(tf.reduce_max(y_pred, axis = -1), tf.minimum(performance_count, tf.shape(y_pred)[0]), sorted = True).indices
        y_pred = tf.gather(y_pred, top_indices)
        bbox_pred = tf.gather(bbox_pred, top_indices)
        if anchors is not None:
            anchors = tf.gather(anchors, top_indices)
        if mask_pred is not None:
            mask_pred = tf.gather(mask_pred, top_indices)
          
    bbox_flag = (tf.keras.backend.ndim(bbox_pred) == 3)
    mask_flag = (mask_pred is not None and tf.keras.backend.int_shape(mask_pred)[-1] != 1)
    if bbox_flag or mask_flag:
        #label_indices = tf.stack([tf.range(tf.shape(bbox_pred)[0]), ], axis = -1)
        label_indices = tf.argmax(y_pred, axis = -1, output_type = tf.int32)
        if bbox_flag:
            #bbox_pred = tf.gather_nd(bbox_pred, label_indices)
            bbox_pred = tf.gather(bbox_pred, label_indices, batch_dims = 1)
        if mask_flag:
            mask_pred = tf.transpose(mask_pred, [0, 3, 1, 2])
            #mask_pred = tf.gather_nd(mask_pred, label_indices)
            mask_pred = tf.gather(mask_pred, label_indices, batch_dims = 1)
            mask_pred = tf.expand_dims(mask_pred, axis = -1)
        
    if anchors is not None and callable(coder_func):
        bbox_pred = coder_func(anchors, bbox_pred, **kwargs)
        bbox_pred = tf.clip_by_value(bbox_pred, 0, 1)
    x1, y1, x2, y2 = tf.split(bbox_pred, 4, axis = -1)
    bbox = tf.concat([y1, x1, y2, x2], axis = -1)
    
    scores = []
    indices = []
    for cls in keep_label:
        threshold = score_threshold[cls]
        score = y_pred[..., cls]
        cls_indices = tf.image.non_max_suppression_with_scores(bbox, score, max_output_size = proposal_count, iou_threshold = iou_threshold, score_threshold = threshold, soft_nms_sigma = soft_nms_sigma)[0]
        scores.append(tf.gather(score, cls_indices))
        indices.append(cls_indices)
    scores = tf.concat(scores, axis = 0)
    indices = tf.concat(indices, axis = 0)
    
    top_indices = tf.nn.top_k(scores, tf.minimum(proposal_count, tf.shape(scores)[0]), sorted = True).indices
    indices = tf.gather(indices, top_indices)
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