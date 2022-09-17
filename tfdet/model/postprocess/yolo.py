import tensorflow as tf

from tfdet.core.bbox import yolo2bbox
from tfdet.core.util import map_fn, multiclass_nms

class FilterDetection(tf.keras.layers.Layer):
    def __init__(self, proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05, soft_nms = False, valid = False, performance_count = 5000,
                 batch_size = 1, clip_ratio = 16 / 1000, **kwargs):
        super(FilterDetection, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.soft_nms = soft_nms
        self.valid = valid
        self.performance_count = performance_count
        self.batch_size = batch_size
        self.clip_ratio = clip_ratio

    def call(self, inputs):
        score, logits, regress, anchors = inputs
        
        if self.valid:
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
        logits = tf.multiply(logits, score)
        anchors = tf.tile(tf.expand_dims(anchors, axis = 0), [tf.shape(logits)[0], 1, 1])
        
        out = map_fn(multiclass_nms, logits, regress, anchors, dtype = (logits.dtype, regress.dtype), batch_size = self.batch_size,
                     proposal_count = self.proposal_count, iou_threshold = self.iou_threshold, score_threshold = self.score_threshold, soft_nms = self.soft_nms, performance_count = self.performance_count,
                     coder_func = yolo2bbox, clip_ratio = self.clip_ratio)
        return out
        
    def get_config(self):
        config = super(FilterDetection, self).get_config()
        config["proposal_count"] = self.proposal_count
        config["iou_threshold"] = self.iou_threshold
        config["score_threshold"] = self.score_threshold
        config["soft_nms"] = self.soft_nms
        config["valid"] = self.valid
        config["performance_count"] = self.performance_count
        config["batch_size"] = self.batch_size
        config["clip_ratio"] = self.clip_ratio
        return config
        