import tensorflow as tf

from tfdet.core.bbox import yolo2bbox
from tfdet.core.util import map_fn

def filter_detection(score, logit, regress, anchors, proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05, soft_nms = False, performance_count = 5000, 
                     clip_ratio = 16 / 1000):
    n_class = tf.keras.backend.int_shape(logit)[-1]
    logit = score * logit
    score = tf.reduce_max(logit, axis = -1, keepdims = True)
    label = tf.argmax(logit, axis = -1)
    valid_indices = tf.where(tf.logical_and(score_threshold <= score, tf.expand_dims(0 < label, axis = -1)))[:, 0]
    logit = tf.gather(logit, valid_indices)
    regress = tf.gather(regress, valid_indices)
    anchors = tf.gather(anchors, valid_indices)
    
    performance_count = tf.minimum(performance_count, tf.shape(logit)[0])
    top_indices = tf.nn.top_k(tf.reduce_max(logit, axis = -1), performance_count, sorted = True).indices
    logit = tf.gather(logit, top_indices)
    regress = tf.gather(regress, top_indices)
    anchors = tf.gather(anchors, top_indices)
    
    regress = yolo2bbox(anchors, regress, clip_ratio)
    regress = tf.clip_by_value(regress, 0, 1)
    x1, y1, x2, y2 = tf.split(regress, 4, axis = -1)
    transfom_regress = tf.concat([y1, x1, y2, x2], axis = -1)
    
    soft_nms_sigma = soft_nms
    if not isinstance(soft_nms, float):
        soft_nms_sigma = 0.5 if soft_nms else 0.
    def _yolo_detection(class_score, label):
        indices = tf.image.non_max_suppression_with_scores(transfom_regress, class_score, max_output_size = proposal_count, iou_threshold = iou_threshold, soft_nms_sigma = soft_nms_sigma)[0]
        label = tf.gather(label, indices)
        indices = tf.stack([tf.cast(indices, label.dtype), label], axis = -1)
        return indices
    
    indices = []
    for c in range(1, n_class):
        class_score = logit[..., c]
        labels = c * tf.ones([tf.shape(class_score)[0]], dtype = tf.int64)
        indices.append(_yolo_detection(class_score, labels))
    indices = tf.concat(indices, axis = 0)

    class_score = tf.gather_nd(logit, indices)
    top_indices = tf.nn.top_k(class_score, tf.minimum(proposal_count, tf.shape(class_score)[0]), sorted = True).indices
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
    def __init__(self, proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05, soft_nms = False, performance_count = 5000, 
                 batch_size = 1, clip_ratio = 16 / 1000, **kwargs):
        super(FilterDetection, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.soft_nms = soft_nms
        self.batch_size = batch_size
        self.clip_ratio = clip_ratio

    def call(self, inputs):
        score, logits, regress, anchors = inputs
        valid_flags = tf.logical_and(tf.less_equal(anchors[..., 2], 1),
                                     tf.logical_and(tf.less_equal(anchors[..., 3], 1),
                                                    tf.logical_and(tf.greater_equal(anchors[..., 0], 0),
                                                                   tf.greater_equal(anchors[..., 1], 0))))
        #valid_indices = tf.range(tf.shape(anchors)[0])[valid_flags]
        valid_indices = tf.where(valid_flags)[:, 0]
        score = tf.gather(score, valid_indices, axis = 1)
        logits = tf.gather(logits, valid_indices, axis = 1)
        regress = tf.gather(regress, valid_indices, axis = 1)
        anchors = tf.gather(anchors, valid_indices)
        anchors = tf.tile(tf.expand_dims(anchors, axis = 0), [tf.shape(logits)[0], 1, 1])
        out = map_fn(filter_detection, score, logits, regress, anchors, dtype = (logits.dtype, regress.dtype), batch_size = self.batch_size,
                     proposal_count = self.proposal_count, iou_threshold = self.iou_threshold, score_threshold = self.score_threshold, soft_nms = self.soft_nms, clip_ratio = self.clip_ratio)
        return out
        
    def get_config(self):
        config = super(FilterDetection, self).get_config()
        config["proposal_count"] = self.proposal_count
        config["iou_threshold"] = self.iou_threshold
        config["score_threshold"] = self.score_threshold
        config["soft_nms"] = self.soft_nms
        config["batch_size"] = self.batch_size
        config["clip_ratio"] = self.clip_ratio
        return config
        