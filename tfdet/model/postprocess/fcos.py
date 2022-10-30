import tensorflow as tf

from tfdet.core.bbox import offset2bbox
from tfdet.core.util import map_fn, multiclass_nms

class FilterDetection(tf.keras.layers.Layer):
    def __init__(self, proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05, soft_nms = False, ignore_label = 0, performance_count = 5000, 
                 batch_size = 1, **kwargs):
        super(FilterDetection, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.soft_nms = soft_nms
        self.ignore_label = ignore_label
        self.performance_count = performance_count
        self.batch_size = batch_size

    def call(self, inputs):
        logits, regress, points = inputs[:3]
        centerness = inputs[3] if 3 < len(inputs) else None
        
        if isinstance(logits, list):
            logits = tf.concat(logits, axis = -2)
            regress = tf.concat(regress, axis = -2)
            points = tf.concat(points, axis = 0)
            if centerness is not None:
                centerness = tf.concat(centerness, axis = -2)
        if centerness is not None:
            logits = tf.multiply(logits, centerness)
            logits = tf.sqrt(logits)
        points = tf.tile(tf.expand_dims(points, axis = 0), [tf.shape(logits)[0], 1, 1])
        
        out = map_fn(multiclass_nms, logits, regress, points, dtype = (logits.dtype, regress.dtype), batch_size = self.batch_size,
                     proposal_count = self.proposal_count, soft_nms = self.soft_nms, iou_threshold = self.iou_threshold, score_threshold = self.score_threshold, ignore_label = self.ignore_label, performance_count = self.performance_count, coder_func = offset2bbox)
        return out
        
    def get_config(self):
        config = super(FilterDetection, self).get_config()
        config["proposal_count"] = self.proposal_count
        config["iou_threshold"] = self.iou_threshold
        config["score_threshold"] = self.score_threshold
        config["soft_nms"] = self.soft_nms
        config["ignore_label"] = self.ignore_label
        config["performance_count"] = self.performance_count
        config["batch_size"] = self.batch_size
        return config