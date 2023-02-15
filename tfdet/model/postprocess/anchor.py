import tensorflow as tf

from tfdet.core.bbox import delta2bbox
from tfdet.core.ops import multiclass_nms
from tfdet.core.util import map_fn

class FilterDetection(tf.keras.layers.Layer):
    def __init__(self, proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05, soft_nms = False, valid_inside_anchor = False, ignore_label = 0, performance_count = 5000,
                 mean = [0., 0., 0., 0.], std = [1., 1., 1., 1.], clip_ratio = 16 / 1000, 
                 batch_size = 1, dtype = tf.float32, 
                 tensorrt = False, **kwargs):
        kwargs["dtype"] = dtype
        super(FilterDetection, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.soft_nms = soft_nms
        self.valid_inside_anchor = valid_inside_anchor
        self.ignore_label = ignore_label
        self.performance_count = performance_count
        self.mean = mean
        self.std = std
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        self.tensorrt = tensorrt

    def call(self, inputs):
        y_pred, bbox_pred, anchors = inputs[:3]
        conf_pred = inputs[3] if 3 < len(inputs) else None
        if isinstance(y_pred, (list, tuple)):
            y_pred = tf.concat(y_pred, axis = -2)
        if isinstance(bbox_pred, (list, tuple)):
            bbox_pred = tf.concat(bbox_pred, axis = -2)
        if isinstance(anchors, (list, tuple)):
            anchors = tf.concat(anchors, axis = 0)
        if conf_pred is not None and isinstance(conf_pred, (list, tuple)):
            conf_pred = tf.concat(conf_pred, axis = -2)
        
        if self.valid_inside_anchor:
            valid_flags = tf.logical_and(tf.less_equal(anchors[..., 2], 1),
                                         tf.logical_and(tf.less_equal(anchors[..., 3], 1),
                                                        tf.logical_and(tf.greater_equal(anchors[..., 0], 0),
                                                                       tf.greater_equal(anchors[..., 1], 0))))
            #valid_indices = tf.range(tf.shape(anchors)[0])[valid_flags]
            valid_indices = tf.where(valid_flags)[:, 0]
            y_pred = tf.gather(y_pred, valid_indices, axis = 1)
            bbox_pred = tf.gather(bbox_pred, valid_indices, axis = 1)
            anchors = tf.gather(anchors, valid_indices)
            if conf_pred is not None:
                conf_pred = tf.gather(conf_pred, valid_indices, axis = 1)
        if conf_pred is not None:
            y_pred = tf.multiply(y_pred, conf_pred)
        
        if not self.tensorrt:
            anchors = tf.tile(tf.expand_dims(anchors, axis = 0), [tf.shape(y_pred)[0], 1, 1])
            out = map_fn(multiclass_nms, y_pred, bbox_pred, anchors, dtype = (self.dtype, self.dtype), batch_size = self.batch_size,
                         proposal_count = self.proposal_count, soft_nms = self.soft_nms, iou_threshold = self.iou_threshold, score_threshold = self.score_threshold, ignore_label = self.ignore_label, performance_count = self.performance_count,
                         coder_func = delta2bbox, mean = self.mean, std = self.std, clip_ratio = self.clip_ratio)
        else:
            bbox_pred = delta2bbox(anchors, bbox_pred, mean = self.mean, std = self.std, clip_ratio = self.clip_ratio)
            bbox_pred = tf.clip_by_value(bbox_pred, 0, 1)
            out = (y_pred, bbox_pred)
        return out
        
    def get_config(self):
        config = super(FilterDetection, self).get_config()
        config["proposal_count"] = self.proposal_count
        config["iou_threshold"] = self.iou_threshold
        config["score_threshold"] = self.score_threshold
        config["soft_nms"] = self.soft_nms
        config["valid_inside_anchor"] = self.valid_inside_anchor
        config["ignore_label"] = self.ignore_label
        config["performance_count"] = self.performance_count
        config["mean"] = self.mean
        config["std"] = self.std
        config["clip_ratio"] = self.clip_ratio
        config["batch_size"] = self.batch_size
        config["tensorrt"] = self.tensorrt
        return config