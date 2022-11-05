import tensorflow as tf
import numpy as np

from tfdet.core.bbox import delta2bbox
from tfdet.core.util import map_fn, multiclass_nms

class FilterDetection(tf.keras.layers.Layer):
    def __init__(self, proposal_count = 100, iou_threshold = 0.5, score_threshold = 0.05, soft_nms = False, ensemble = True, valid = False, ignore_label = 0, performance_count = 5000,
                 batch_size = 1, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000,
                 tensorrt = False, **kwargs):
        super(FilterDetection, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.soft_nms = soft_nms
        self.ensemble = ensemble
        self.valid = valid
        self.ignore_label = ignore_label
        self.performance_count = performance_count
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.clip_ratio = clip_ratio
        self.tensorrt = tensorrt

    def call(self, inputs):
        if 5 < len(inputs):
            inputs = inputs[3:7]  
        cls_logits, cls_regress, proposals = inputs[:3]
        mask_regress = inputs[3] if 3 < len(inputs) else None
        
        std = self.std
        if isinstance(proposals, list):
            proposals = proposals[-1]
        if isinstance(cls_logits, list):
            std = np.divide(std, len(cls_logits))
            cls_logits, cls_regress = tf.reduce_mean(cls_logits, axis = 0) if self.ensemble else cls_logits[-1], cls_regress[-1]
            if mask_regress is not None:
                mask_regress = tf.reduce_mean(mask_regress, axis = 0) if self.ensemble else mask_regress[-1]
        elif tf.keras.backend.int_shape(cls_logits)[-1] == 1: #rpn_score, rpn_regress, anchors
            cls_logits = tf.concat([1 - cls_logits, cls_logits], axis = -1)
            if self.valid:
                valid_flags = tf.logical_and(tf.less_equal(proposals[..., 2], 1),
                                             tf.logical_and(tf.less_equal(proposals[..., 3], 1),
                                                            tf.logical_and(tf.greater_equal(proposals[..., 0], 0),
                                                                           tf.greater_equal(proposals[..., 1], 0))))
                #valid_indices = tf.range(tf.shape(proposals)[0])[valid_flags]
                valid_indices = tf.where(valid_flags)[:, 0]
                cls_logits = tf.gather(cls_logits, valid_indices, axis = 1)
                cls_regress = tf.gather(cls_regress, valid_indices, axis = 1)
                proposals = tf.gather(proposals, valid_indices)
            if not self.tensorrt:
                proposals = tf.tile(tf.expand_dims(proposals, axis = 0), [tf.shape(cls_logits)[0], 1, 1])
        
        dtype = (cls_logits.dtype, cls_regress.dtype)
        if mask_regress is not None:
            dtype += (mask_regress.dtype,)
        args = [l for l in [cls_logits, cls_regress, proposals, mask_regress] if l is not None]
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
        config["valid"] = self.valid
        config["ignore_label"] = self.ignore_label
        config["performance_count"] = self.performance_count
        config["batch_size"] = self.batch_size
        config["mean"] = self.mean
        config["std"] = self.std
        config["clip_ratio"] = self.clip_ratio
        config["tensorrt"] = self.tensorrt
        return config