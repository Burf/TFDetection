import tensorflow as tf
import numpy as np

from tfdet.core.bbox import delta2bbox
from tfdet.core.ops import multiclass_nms
from tfdet.core.util import map_fn

class FilterDetection(tf.keras.layers.Layer):
    def __init__(self, proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05, soft_nms = False, ensemble = True, valid_inside_anchor = False, ignore_label = 0, performance_count = 5000,
                 mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000,
                 batch_size = 1, dtype = tf.float32,
                 tensorrt = False, **kwargs):
        kwargs["dtype"] = dtype
        super(FilterDetection, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.soft_nms = soft_nms
        self.ensemble = ensemble
        self.valid_inside_anchor = valid_inside_anchor
        self.ignore_label = ignore_label
        self.performance_count = performance_count
        self.mean = mean
        self.std = std
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        self.tensorrt = tensorrt

    def call(self, inputs):
        if 5 < len(inputs):
            inputs = inputs[3:]
        y_pred, bbox_pred, proposals = inputs[:3]
        mask_pred = inputs[3] if 3 < len(inputs) else None
        if mask_pred is not None and tf.keras.backend.ndim(mask_pred[0] if isinstance(mask_pred, (tuple, list)) else mask_pred) != 5:
            mask_pred = None
        
        n_class = tf.keras.backend.int_shape(y_pred[0] if isinstance(y_pred, (tuple, list)) else y_pred)[-1]
        std = self.std
        if 1 < n_class: #2-stage cls
            if isinstance(proposals, (list, tuple)):
                proposals = proposals[-1]
            if isinstance(y_pred, (list, tuple)):
                std = np.divide(std, len(y_pred))
                y_pred, bbox_pred = tf.reduce_mean(y_pred, axis = 0) if self.ensemble else y_pred[-1], bbox_pred[-1]
                if mask_pred is not None:
                    mask_pred = tf.reduce_mean(mask_pred, axis = 0) if self.ensemble else mask_pred[-1]
        else: #1-stage rpn
            if isinstance(y_pred, (list, tuple)):
                y_pred = tf.concat(y_pred, axis = -2)
            if isinstance(bbox_pred, (list, tuple)):
                bbox_pred = tf.concat(bbox_pred, axis = -2)
            if isinstance(proposals, (list, tuple)):
                proposals = tf.concat(proposals, axis = 0)
            #y_pred = tf.concat([1 - y_pred, y_pred], axis = -1)
            if self.valid_inside_anchor:
                valid_flags = tf.logical_and(tf.less_equal(proposals[..., 2], 1),
                                             tf.logical_and(tf.less_equal(proposals[..., 3], 1),
                                                            tf.logical_and(tf.greater_equal(proposals[..., 0], 0),
                                                                           tf.greater_equal(proposals[..., 1], 0))))
                #valid_indices = tf.range(tf.shape(proposals)[0])[valid_flags]
                valid_indices = tf.where(valid_flags)[:, 0]
                y_pred = tf.gather(y_pred, valid_indices, axis = 1)
                bbox_pred = tf.gather(bbox_pred, valid_indices, axis = 1)
                proposals = tf.gather(proposals, valid_indices)
            if not self.tensorrt:
                proposals = tf.tile(tf.expand_dims(proposals, axis = 0), [tf.shape(y_pred)[0], 1, 1])
                
        dtype = (self.dtype, self.dtype)
        if mask_pred is not None:
            dtype  = (self.dtype, self.dtype, self.dtype)
        args = [l for l in [y_pred, bbox_pred, proposals, mask_pred] if l is not None]
        if not self.tensorrt:
            out = map_fn(multiclass_nms, *args, dtype = dtype, batch_size = self.batch_size, 
                         proposal_count = self.proposal_count, iou_threshold = self.iou_threshold, score_threshold = self.score_threshold, soft_nms = self.soft_nms, ignore_label = self.ignore_label, performance_count = self.performance_count,
                         coder_func = delta2bbox, mean = self.mean, std = std, clip_ratio = self.clip_ratio)
        else:
            raise ValueError("Conversion of rcnn is not yet supported.")
        return out
        
    def get_config(self):
        config = super(FilterDetection, self).get_config()
        config["proposal_count"] = self.proposal_count
        config["iou_threshold"] = self.iou_threshold
        config["score_threshold"] = self.score_threshold
        config["soft_nms"] = self.soft_nms
        config["ensemble"] = self.ensemble
        config["valid_inside_anchor"] = self.valid_inside_anchor
        config["ignore_label"] = self.ignore_label
        config["performance_count"] = self.performance_count
        config["mean"] = self.mean
        config["std"] = self.std
        config["clip_ratio"] = self.clip_ratio
        config["batch_size"] = self.batch_size
        config["tensorrt"] = self.tensorrt
        return config