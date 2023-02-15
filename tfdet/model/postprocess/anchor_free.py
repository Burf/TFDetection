import tensorflow as tf

from tfdet.core.bbox import offset2bbox
from tfdet.core.ops import multiclass_nms
from tfdet.core.util import map_fn

class FilterDetection(tf.keras.layers.Layer):
    def __init__(self, proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05, soft_nms = False, ignore_label = 0, performance_count = 5000, 
                 batch_size = 1, dtype = tf.float32,
                 tensorrt = False, **kwargs):
        kwargs["dtype"] = dtype
        super(FilterDetection, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.soft_nms = soft_nms
        self.ignore_label = ignore_label
        self.performance_count = performance_count
        self.batch_size = batch_size
        self.tensorrt = tensorrt

    def call(self, inputs):
        y_pred, bbox_pred, points = inputs[:3]
        conf_pred = inputs[3] if 3 < len(inputs) else None
        if isinstance(y_pred, (list, tuple)):
            y_pred = tf.concat(y_pred, axis = -2)
        if isinstance(bbox_pred, (list, tuple)):
            bbox_pred = tf.concat(bbox_pred, axis = -2)
        if isinstance(points, (list, tuple)):
            points = tf.concat(points, axis = 0)
        if conf_pred is not None:
            if isinstance(conf_pred, (list, tuple)):
                conf_pred = tf.concat(conf_pred, axis = -2)
            y_pred = tf.multiply(y_pred, conf_pred)
            y_pred = tf.sqrt(y_pred)
            
        if not self.tensorrt:
            points = tf.tile(tf.expand_dims(points, axis = 0), [tf.shape(y_pred)[0], 1, 1])
            out = map_fn(multiclass_nms, y_pred, bbox_pred, points, dtype = (self.dtype, self.dtype), batch_size = self.batch_size,
                         proposal_count = self.proposal_count, soft_nms = self.soft_nms, iou_threshold = self.iou_threshold, score_threshold = self.score_threshold, ignore_label = self.ignore_label, performance_count = self.performance_count, coder_func = offset2bbox)
        else:
            bbox_pred = offset2bbox(points, bbox_pred)
            bbox_pred = tf.clip_by_value(bbox_pred, 0, 1)
            out = (y_pred, bbox_pred)
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
        config["tensorrt"] = self.tensorrt
        return config