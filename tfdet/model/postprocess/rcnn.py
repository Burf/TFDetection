import tensorflow as tf

from tfdet.core.bbox import delta2bbox
from tfdet.core.util import map_fn

def filter_detection(cls_logit, cls_regress, proposal, mask_regress = None, proposal_count = 100, iou_threshold = 0.3, score_threshold = 0.7, soft_nms = False, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000):
    """
    cls_logit = classifier logit #(num_proposals, num_class)
    cls_regress = classifier regress #(num_proposals, num_class, delta)
    mask_regress = mask regress #(num_proposals, h, w, num_class)
    proposal = proposal anchor #(num_proposals, bbox)

    logit = targeted logit #(proposal_count, num_class)
    proposal = [[x1, y1, x2, y2], ...] #(proposal_count, bbox)
    mask = targeted mask #(proposal_count, h, w)
    """
    valid_indices = tf.where(0 < tf.reduce_max(proposal, axis = -1))
    cls_logit = tf.gather_nd(cls_logit, valid_indices)
    cls_regress = tf.gather_nd(cls_regress, valid_indices)
    proposal = tf.gather_nd(proposal, valid_indices)
    if mask_regress is not None:
        mask_regress = tf.gather_nd(mask_regress, valid_indices)
    
    n_class = tf.keras.backend.int_shape(cls_logit)[-1]
    logit_indices = tf.argmax(cls_logit, axis = -1, output_type = tf.int32)
    indices = tf.stack([tf.range(tf.shape(cls_logit)[0]), logit_indices], axis = -1)
    score = tf.gather_nd(cls_logit, indices)
    delta = tf.gather_nd(cls_regress, indices)
    if mask_regress is not None:
        h, w = tf.keras.backend.int_shape(mask_regress)[-3:-1]
        mask = tf.transpose(mask_regress, [0, 3, 1, 2])
        mask = tf.gather_nd(mask, indices)

    # Transform delta to bbox
    proposal = delta2bbox(proposal, delta, mean, std, clip_ratio)

    # Clipping to valid area
    proposal = tf.clip_by_value(proposal, 0, 1)
    
    # Transform
    x1, y1, x2, y2 = tf.split(proposal, 4, axis = -1)
    transform_proposal = tf.concat([y1, x1, y2, x2], axis = -1)

    # Filter out background boxes and score threshold
    keep = tf.where(0 < logit_indices)
    score_keep = tf.where(score_threshold <= score)
    keep = tf.sets.intersection(tf.expand_dims(keep[:, 0], 0), tf.expand_dims(score_keep[:, 0], 0))
    keep = tf.expand_dims(tf.sparse.to_dense(keep)[0], axis = -1)
    nms_indices = tf.gather_nd(logit_indices, keep)
    nms_score = tf.gather_nd(score, keep)
    nms_proposal = tf.gather_nd(transform_proposal, keep)
    unique_nms_class = tf.unique(nms_indices)[0]

    # Map over class
    soft_nms_sigma = soft_nms
    if not isinstance(soft_nms, float):
        soft_nms_sigma = 0.5 if soft_nms else 0.
    def unique_nms_keep(cls):
        unique_nms_indices = tf.where(tf.equal(nms_indices, cls))
        unique_nms_keep = tf.image.non_max_suppression_with_scores(tf.gather_nd(nms_proposal, unique_nms_indices), tf.gather_nd(nms_score, unique_nms_indices), max_output_size = proposal_count, iou_threshold = iou_threshold, soft_nms_sigma = soft_nms_sigma)[0]
        unique_nms_keep = tf.gather_nd(keep, tf.gather(unique_nms_indices, unique_nms_keep))
        pad_count = tf.maximum(proposal_count - tf.shape(unique_nms_keep)[0], 0)
        unique_nms_keep = tf.pad(unique_nms_keep, [[0, pad_count], [0, 0]])
        unique_nms_keep = tf.reshape(unique_nms_keep, [proposal_count])
        return unique_nms_keep
    nms_keep = map_fn(unique_nms_keep, unique_nms_class, dtype = tf.int64)
    nms_keep = tf.reshape(nms_keep, (-1,))
    nms_indices = tf.where(-1 < nms_keep)
    nms_keep = tf.gather_nd(nms_keep, nms_indices)
    
    # Compute intersection
    keep = tf.sets.intersection(tf.expand_dims(keep[:, 0], 0), tf.expand_dims(nms_keep, 0))
    keep = tf.sparse.to_dense(keep)[0]

    # Keep top detections
    score_keep = tf.gather(score, keep)
    n_keep = tf.minimum(tf.shape(score_keep)[0], proposal_count)
    top_indices = tf.nn.top_k(score_keep, k = n_keep)[1]
    keep = tf.gather(keep, top_indices)

    # Slicing
    logit = tf.gather(cls_logit, keep)
    proposal = tf.gather(proposal, keep)
    if mask_regress is not None:
        mask = tf.gather(mask, keep)
    
    # Padding
    pad_count = tf.maximum(proposal_count - tf.shape(proposal)[0], 0)
    logit = tf.pad(logit, [[0, pad_count], [0, 0]])
    proposal = tf.pad(proposal, [[0, pad_count], [0, 0]])
    logit = tf.reshape(logit, [proposal_count, n_class])
    proposal = tf.reshape(proposal, [proposal_count, 4])
    result = logit, proposal
    if mask_regress is not None:
        mask = tf.pad(mask, [[0, pad_count], [0, 0], [0, 0]])
        mask = tf.reshape(mask, [proposal_count, h, w])
        result = logit, proposal, mask
    return result

class FilterDetection(tf.keras.layers.Layer):
    def __init__(self, proposal_count = 100, iou_threshold = 0.3, score_threshold = 0.7, soft_nms = False, ensemble = True, batch_size = 1, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, **kwargs):
        super(FilterDetection, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.soft_nms = soft_nms
        self.ensemble = ensemble
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.clip_ratio = clip_ratio

    def call(self, inputs):
        if 5 < len(inputs):
            inputs = inputs[3:7]  
        cls_logits, cls_regress, proposals = inputs[:3]
        mask_regress = inputs[3] if 3 < len(inputs) else None
        if isinstance(proposals, list):
            proposals = proposals[-1]
        if isinstance(cls_logits, list):
            cls_logits, cls_regress = tf.reduce_mean(cls_logits, axis = 0) if self.ensemble else cls_logits[-1], cls_regress[-1]
            if mask_regress is not None:
                mask_regress = tf.reduce_mean(mask_regress, axis = 0) if self.ensemble else mask_regress[-1]
        dtype = (cls_logits.dtype, cls_regress.dtype)
        if mask_regress is not None:
            dtype += (mask_regress.dtype,)
        args = [l for l in [cls_logits, cls_regress, proposals, mask_regress] if l is not None]
        out = map_fn(filter_detection, *args, dtype = dtype, batch_size = self.batch_size, 
                     proposal_count = self.proposal_count, iou_threshold = self.iou_threshold, score_threshold = self.score_threshold, soft_nms = self.soft_nms, mean = self.mean, std = self.std, clip_ratio = self.clip_ratio)
        return out
        
    def get_config(self):
        config = super(FilterDetection, self).get_config()
        config["proposal_count"] = self.proposal_count
        config["iou_threshold"] = self.iou_threshold
        config["score_threshold"] = self.score_threshold
        config["soft_nms"] = self.soft_nms
        config["ensemble"] = self.ensemble
        config["batch_size"] = self.batch_size
        config["mean"] = self.mean
        config["std"] = self.std
        config["clip_ratio"] = self.clip_ratio
        return config