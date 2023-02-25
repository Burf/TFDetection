import functools

import tensorflow as tf

from tfdet.core.assign import point
from tfdet.core.bbox import bbox2offset, offset2bbox, offset2centerness
from tfdet.core.loss import binary_cross_entropy, focal_binary_cross_entropy, iou, weight_reduce_loss
from tfdet.core.util import map_fn
from .util import image_to_level

def focal_loss(y_true, y_pred, alpha = .25, gamma = 2., weight = None, reduce = tf.reduce_mean):
    return focal_binary_cross_entropy(y_true, y_pred, alpha = alpha, gamma = gamma, weight = weight, reduce = reduce)

class AnchorFreeLoss(tf.keras.layers.Layer):
    def __init__(self, class_loss = focal_loss, bbox_loss = iou, conf_loss = binary_cross_entropy,
                 decode_bbox = True, weight = None, background = False,
                 assign = point, sampler = None,
                 batch_size = 1,
                 missing_value = 0., dtype = tf.float32, **kwargs):
        kwargs["dtype"] = dtype
        super(AnchorFreeLoss, self).__init__(**kwargs)
        self.class_loss = class_loss
        self.bbox_loss = bbox_loss
        self.conf_loss = conf_loss
        self.decode_bbox = decode_bbox
        self.weight = weight
        self.background = background
        self.assign = assign
        self.sampler = sampler
        self.batch_size = batch_size
        self.missing_value = missing_value
        
        target = functools.partial(self.target, assign = self.assign, sampler = self.sampler, decode_bbox = self.decode_bbox)
        loss = functools.partial(self.loss, class_loss = self.class_loss, bbox_loss = self.bbox_loss, conf_loss = self.conf_loss, sampling = self.sampler is not None, weight = self.weight, background = self.background, missing_value = self.missing_value)
        target_eval = functools.partial(self.target, assign = self.assign, sampler = None, decode_bbox = self.decode_bbox)
        loss_eval = functools.partial(self.loss, class_loss = self.class_loss, bbox_loss = self.bbox_loss, conf_loss = self.conf_loss, sampling = False, weight = self.weight, background = self.background, missing_value = self.missing_value)
        dtype = (tf.int8, self.dtype, self.dtype)
        conf_dtype = (tf.int8, self.dtype, self.dtype, self.dtype)
        self.target_func = tf.keras.layers.Lambda(lambda args: map_fn(target, *args, dtype = (conf_dtype if 4 < len(args) and tf.keras.backend.int_shape(args[-1])[-1] == 1 else dtype), batch_size = self.batch_size), dtype = self.dtype, name = "target")
        self.loss_func = tf.keras.layers.Lambda(lambda args: loss(*args), dtype = self.dtype, name = "loss")
        self.target_eval_func = tf.keras.layers.Lambda(lambda args: map_fn(target_eval, *args, dtype = (conf_dtype if 4 < len(args) and tf.keras.backend.int_shape(args[-1])[-1] == 1 else dtype), batch_size = self.batch_size), dtype = self.dtype, name = "target_eval")
        self.loss_eval_func = tf.keras.layers.Lambda(lambda args: loss_eval(*args), dtype = self.dtype, name = "loss_eval")
        
    @staticmethod
    @tf.function
    def target(y_true, bbox_true, y_pred, points, regress_range = None, conf_pred = None, assign = point, sampler = None, decode_bbox = True):
        """
        Args:
            y_true = label #(padded_num_true, 1 or num_class)
            bbox_true = [[x1, y1, x2, y2], ...] #(padded_num_true, 4)
            y_pred = classifier logit #(num_points, num_class)
            points = [[x1, y1, x2, y2], ...] #(num_points, 2)
            regress_range = [[min_offset_range, max_offet_range], ...] #(num_points, 2) (optional)
            conf_pred = classifier confidence score #(num_points, 1) (optional)

        Returns:
            state = -1 : negative / 0 : neutral / 1 : positive #(num_points, 1)
            y_true = label #(num_points, 1 or num_class)
            bbox_true = [[x1, y1, x2, y2], ...] #(num_points, 4)
            conf_true = confidence score #(num_points, 1) (optional)
        """
        if regress_range is not None and tf.keras.backend.int_shape(regress_range)[-1] == 1:
            conf_pred = regress_range
            regress_range = None
            
        valid_indices = tf.where(tf.reduce_any(tf.greater(bbox_true, 0), axis = -1))[:, 0]
        bbox_true = tf.gather(bbox_true, valid_indices)
        if tf.keras.backend.int_shape(y_pred)[-1] == 1:
            y_true = tf.ones([tf.shape(bbox_true)[0], 1], dtype = tf.float32)
        else:
            y_true = tf.gather(y_true, valid_indices)
        
        if conf_pred is not None:
            y_pred = tf.multiply(y_pred, conf_pred)
            y_pred = tf.sqrt(y_pred)
        
        if regress_range is not None:
            true_indices, positive_indices, negative_indices = assign(y_true, bbox_true, y_pred, points, regress_range = regress_range)
        else:
            true_indices, positive_indices, negative_indices = assign(y_true, bbox_true, y_pred, points)
        if sampler is not None:
            true_indices, positive_indices, negative_indices = sampler(true_indices, positive_indices, negative_indices)

        _positive_indices = tf.expand_dims(positive_indices, axis = -1)
        negative_indices = tf.expand_dims(negative_indices, axis = -1)

        pred_count = tf.shape(points)[0]
        state = tf.zeros([pred_count, 1], dtype = tf.int8)
        positive_state = tf.tensor_scatter_nd_update(state, _positive_indices, tf.ones_like(_positive_indices, dtype = tf.int8))
        state = tf.tensor_scatter_nd_update(positive_state, negative_indices, -tf.ones_like(negative_indices, dtype = tf.int8))

        n_class = tf.shape(y_true)[-1]
        _y_true = tf.cond(tf.equal(n_class, 1), true_fn = lambda: tf.zeros([pred_count, 1], dtype = tf.float32), false_fn = lambda: tf.pad(tf.ones([pred_count, 1], dtype = tf.float32), [[0, 0], [0, n_class - 1]]))
        y_true = tf.gather(y_true, true_indices)
        _y_true = tf.tensor_scatter_nd_update(_y_true, _positive_indices, tf.cast(y_true, tf.float32))
        _bbox_true = tf.zeros([pred_count, 4], dtype = tf.float32)
        if conf_pred is not None:
            conf_true = tf.zeros([pred_count, 1], dtype = tf.float32)
        if tf.keras.backend.int_shape(true_indices)[0] != 0:
            bbox_true = tf.gather(bbox_true, true_indices)
            if not decode_bbox:
                points = tf.gather(points, positive_indices)
                bbox_true = bbox2offset(bbox_true, points)
            _bbox_true = tf.tensor_scatter_nd_update(_bbox_true, _positive_indices, bbox_true)
            if conf_pred is not None:
                if decode_bbox:
                    points = tf.gather(points, positive_indices)
                    bbox_true = bbox2offset(bbox_true, points)
                _conf_true = offset2centerness(bbox_true)
                conf_true = tf.tensor_scatter_nd_update(conf_true, _positive_indices, _conf_true)
        if conf_pred is not None:
            return state, _y_true, _bbox_true, conf_true
        else:
            return state, _y_true, _bbox_true
    
    @staticmethod
    @tf.function
    def loss(state_list, y_true_list, bbox_true_list, y_pred_list, bbox_pred_list, conf_true_list = None, conf_pred_list = None, class_loss = focal_loss, bbox_loss = iou, conf_loss = binary_cross_entropy, sampling = False, weight = None, background = False, missing_value = 0.):
        """
        Args:
            state_list = level state list #[(N, num_anchors_1, 1), ..., (N, num_anchors_n, 1)]
            y_true_list = level label list #[(N, num_anchors_1, 1 or num_class), ..., (N, num_anchors_n, 1 or num_class)]
            bbox_true_list = level bbox_true list #[(N, num_anchors_1, 4), ..., (N, num_anchors_n, 4)]
            y_pred_list = level logits list #[(N, num_anchors_1, num_class), ..., (N, num_anchors_n, num_class)]
            bbox_pred_list = level bbox_pred list #[(N, num_anchors_1, 4), ..., (N, num_anchors_n, 4)]
            conf_true_list = level score_true list #[(N, num_anchors_1, 1), ..., (N, num_anchors_n, 1)] (optional)
            conf_pred_list = level score_pred list #[(N, num_anchors_1, 1), ..., (N, num_anchors_n, 1)] (optional)

        Returns:
            class_loss_list = level class loss #[(1,), ..., (1,)]
            bbox_loss_list = level bbox loss #[(1,), ..., (1,)]
            conf_loss_list = level confidence score loss #[(1,), ..., (1,)] (optional)
        """
        class_loss_list = []
        bbox_loss_list = []
        conf_loss_list = []
        for i in range(len(state_list)):
            state, y_true, bbox_true = state_list[i], y_true_list[i], bbox_true_list[i]
            y_pred, bbox_pred = y_pred_list[i], bbox_pred_list[i]
            if conf_true_list is not None and conf_pred_list is not None:
                conf_true, conf_pred = conf_true_list[i], conf_pred_list[i]
            
            dtype = y_pred.dtype
            n_true_class = tf.shape(y_true)[-1]
            n_pred_class = tf.shape(y_pred)[-1]
            
            positive_weight = tf.cast(tf.equal(state, 1), dtype)
            target_weight = tf.cast(tf.not_equal(state, 0), dtype)
            if sampling: #pos + neg
                avg_factor = tf.reduce_sum(tf.maximum(tf.reduce_sum(target_weight, axis = -2), tf.cast(1., dtype)))
            else: #pos
                avg_factor = tf.reduce_sum(tf.maximum(tf.reduce_sum(positive_weight, axis = -2), tf.cast(1., dtype)))
                
            y_true = tf.cond(tf.logical_and(tf.equal(n_true_class, 1), tf.not_equal(n_pred_class, 1)), true_fn = lambda: tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), n_pred_class)[..., 0, :], dtype), false_fn = lambda: tf.cast(y_true, dtype))
            if not background:
                y_true = y_true * positive_weight
            
            _class_loss = class_loss(y_true, y_pred, weight = weight, reduce = None)
            _class_loss = weight_reduce_loss(_class_loss, weight = target_weight, avg_factor = avg_factor)
            
            bbox_positive_weight = positive_weight
            bbox_avg_factor = avg_factor
            if conf_true_list is not None and conf_pred_list is not None:
                conf_true = tf.cast(conf_true, dtype)
                bbox_positive_weight = conf_true
                bbox_avg_factor = tf.reduce_sum(tf.maximum(tf.reduce_sum(conf_true, axis = -2), tf.cast(1e-6, dtype)))
            _bbox_loss = bbox_loss(bbox_true, bbox_pred, reduce = None)
            _bbox_loss = weight_reduce_loss(_bbox_loss, weight = bbox_positive_weight, avg_factor = bbox_avg_factor)
            
            _class_loss = tf.where(tf.logical_or(tf.math.is_nan(_class_loss), tf.math.is_inf(_class_loss)), tf.cast(missing_value, dtype), _class_loss)
            _bbox_loss = tf.where(tf.logical_or(tf.math.is_nan(_bbox_loss), tf.math.is_inf(_bbox_loss)), tf.cast(missing_value, dtype), _bbox_loss)
            class_loss_list.append(_class_loss)
            bbox_loss_list.append(_bbox_loss)
            if conf_true_list is not None and conf_pred_list is not None:
                _conf_loss = conf_loss(conf_true, conf_pred, reduce = None)
                _conf_loss = weight_reduce_loss(_conf_loss, weight = positive_weight, avg_factor = avg_factor)
                
                _conf_loss = tf.where(tf.logical_or(tf.math.is_nan(_conf_loss), tf.math.is_inf(_conf_loss)), tf.cast(missing_value, dtype), _conf_loss)
                conf_loss_list.append(_conf_loss)
                
        if len(class_loss_list) == 1:
            class_loss_list = class_loss_list[0]
        if len(bbox_loss_list) == 1:
            bbox_loss_list = bbox_loss_list[0]
        if 0 < len(conf_loss_list):
            if len(conf_loss_list) == 1:
                conf_loss_list = conf_loss_list[0]
            return class_loss_list, bbox_loss_list, conf_loss_list
        else:
            return class_loss_list, bbox_loss_list
    
    def call(self, inputs, outputs, training = None):
        y_true, bbox_true = inputs
        y_pred, bbox_pred, points = outputs[:3]
        regress_range = conf = None
        if 3 < len(outputs):
            regress_range = outputs[3]
        if 4 < len(outputs):
            conf_pred = outputs[4]
        if not isinstance(y_pred, (tuple, list)):
            y_pred = [y_pred]
        if not isinstance(bbox_pred, (tuple, list)):
            bbox_pred = [bbox_pred]
        if not isinstance(points, (tuple, list)):
            points = [points]
        if regress_range is not None and not isinstance(regress_range, (tuple, list)):
            regress_range = [regress_range]
        if conf_pred is not None and not isinstance(conf_pred, (tuple, list)):
            conf_pred = [conf_pred]
        if regress_range is not None and tf.keras.backend.int_shape(regress_range[0])[-1] == 1:
            conf_pred = regress_range
            regress_range = None
            
        concat_y_pred = tf.concat(y_pred, axis = -2)
        concat_points = tf.tile(tf.expand_dims(tf.concat(points, axis = 0), axis = 0), [tf.shape(bbox_true)[0], 1, 1])
        concat_regress_range = tf.tile(tf.expand_dims(tf.concat(regress_range, axis = 0), axis = 0), [tf.shape(bbox_true)[0], 1, 1]) if regress_range is not None else None
        concat_conf_pred = tf.concat(conf_pred, axis = -2) if conf_pred is not None else None
        args = [arg for arg in [y_true, bbox_true, concat_y_pred, concat_points, concat_regress_range, concat_conf_pred] if arg is not None]
        if training:
            out = self.target_func(args)
        else:
            out = self.target_eval_func(args)
        state, y_true, bbox_true = out[:3]
        conf_true = out[3] if 3 < len(out) else None
        _bbox_pred = bbox_pred
        if self.decode_bbox:
            _bbox_pred = [offset2bbox(points[i], bbox_pred[i]) for i in range(len(points))]
    
        n_level = [tf.shape(point)[0] for point in points]
        state = image_to_level(state, n_level)
        y_true = image_to_level(y_true, n_level)
        bbox_true = image_to_level(bbox_true, n_level)
        if conf_true is not None:
            conf_true = image_to_level(conf_true, n_level)
        
        args = [arg for arg in [state, y_true, bbox_true, y_pred, _bbox_pred, conf_true, conf_pred] if arg is not None]
        if training:
            out = self.loss_func(args)
        else:
            out = self.loss_eval_func(args)
        return out #class_loss, bbox_loss or class_loss, bbox_loss, conf_loss