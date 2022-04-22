import tensorflow as tf

from tfdet.core.bbox import delta2bbox, offset2bbox
from tfdet.core.util import map_fn

def filter_detection(logit, regress, anchors, centerness = None, proposal_count = 100, iou_threshold = 0.3, score_threshold = 0.7, nms = True, soft_nms = False, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000):
    """
    logit = classifier logit #(num_anchors, num_class)
    regress = classifier regress #(num_anchors, delta)
    anchors = anchors or points #(num_anchors, 4) or (num_anchors, 2)
    centerness = classifier centerness #(num_boxes, 1)

    logit = target logit #(proposal_count, num_class)
    proposal = [[x1, y1, x2, y2], ...] #(proposal_count, bbox)
    """
    n_class = tf.keras.backend.int_shape(logit)[-1]
    label = tf.argmax(logit, axis = -1)
    indices = tf.where(0 < label)
    logit = tf.gather_nd(logit, indices)
    regress = tf.gather_nd(regress, indices)
    anchors = tf.gather_nd(anchors, indices)
    if centerness is not None:
        centerness = tf.gather_nd(centerness, indices)
        
    if tf.keras.backend.int_shape(anchors)[-1] == 4: #anchors
        regress = delta2bbox(anchors, regress, mean, std, clip_ratio)
    else: #points
        regress = offset2bbox(anchors, regress)
    regress = tf.clip_by_value(regress, 0, 1)
    x1, y1, x2, y2 = tf.split(regress, 4, axis = -1)
    transfom_regress = tf.concat([y1, x1, y2, x2], axis = -1)

    soft_nms_sigma = soft_nms
    if not isinstance(soft_nms, float):
        soft_nms_sigma = 0.5 if soft_nms else 0.
    def _filter_detection(score, label):
        indices = tf.where(score_threshold <= score)
        if nms:
            filtered_score = tf.gather_nd(score, indices)
            filtered_regress = tf.gather_nd(transfom_regress, indices)
            if centerness is not None:
                filtered_centerness = tf.squeeze(tf.gather_nd(centerness, indices), axis = -1)
                filtered_score = tf.sqrt(filtered_score * filtered_centerness)
            nms_indices = tf.image.non_max_suppression_with_scores(filtered_regress, filtered_score, max_output_size = proposal_count, iou_threshold = iou_threshold, soft_nms_sigma = soft_nms_sigma)[0]
            indices = tf.gather(indices, nms_indices)
        label = tf.gather_nd(label, indices)
        indices = tf.stack([indices[:, 0], label], axis = -1)
        return indices
    
    #score = tf.reduce_max(logit, axis = -1)
    #indices = _filter_detection(score, label)
    indices = []
    for c in range(1, n_class):
        score = logit[..., c]
        labels = c * tf.ones([tf.shape(score)[0]], dtype = tf.int64)
        indices.append(_filter_detection(score, labels))
    indices = tf.concat(indices, axis = 0)
    
    score = tf.gather_nd(logit, indices)
    top_indices = tf.nn.top_k(score, tf.minimum(proposal_count, tf.shape(score)[0]), sorted = True).indices
    indices = tf.gather(indices[:, 0], top_indices)
    logit = tf.gather(logit, indices)
    proposal = tf.gather(regress, indices)
    
    pad_count = tf.maximum(proposal_count - tf.shape(proposal)[0], 0)
    logit = tf.pad(logit, [[0, pad_count], [0, 0]])
    proposal = tf.pad(proposal, [[0, pad_count], [0, 0]])
    
    logit = tf.reshape(logit, [proposal_count, n_class])
    proposal = tf.reshape(proposal, [proposal_count, 4])
    return logit, proposal

class FilterDetection(tf.keras.layers.Layer):
    def __init__(self, proposal_count = 100, iou_threshold = 0.3, score_threshold = 0.7, nms = True, soft_nms = False, batch_size = 1, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, **kwargs):
        super(FilterDetection, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.nms = nms
        self.soft_nms = soft_nms
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.clip_ratio = clip_ratio

    def call(self, inputs):
        logits, regress, anchors = inputs[:3]
        centerness = inputs[3] if 3 < len(inputs) else None
        if isinstance(logits, list):
            logits = tf.concat(logits, axis = -2)
            regress = tf.concat(regress, axis = -2)
            anchors = tf.concat(anchors, axis = 0)
            if centerness is not None:
                centerness = tf.concat(centerness, axis = -2)
        anchors = tf.tile(tf.expand_dims(anchors, axis = 0), [tf.shape(logits)[0], 1, 1])
        args = [l for l in [logits, regress, anchors, centerness] if l is not None]
        out = map_fn(filter_detection, *args, dtype = (logits.dtype, regress.dtype), batch_size = self.batch_size,
                     proposal_count = self.proposal_count, nms = self.nms, soft_nms = self.soft_nms, iou_threshold = self.iou_threshold, score_threshold = self.score_threshold, mean = self.mean, std = self.std, clip_ratio = self.clip_ratio)
        return out
        
    def get_config(self):
        config = super(FilterDetection, self).get_config()
        config["proposal_count"] = self.proposal_count
        config["iou_threshold"] = self.iou_threshold
        config["score_threshold"] = self.score_threshold
        config["nms"] = self.nms
        config["soft_nms"] = self.soft_nms
        config["batch_size"] = self.batch_size
        config["mean"] = self.mean
        config["std"] = self.std
        config["clip_ratio"] = self.clip_ratio
        return config