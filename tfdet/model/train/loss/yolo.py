import functools

import tensorflow as tf

from tfdet.core.assign import max_iou
from tfdet.core.bbox import bbox2yolo, yolo2bbox
from tfdet.core.loss import binary_cross_entropy, focal_binary_cross_entropy, ciou, weight_reduce_loss
from tfdet.core.util import map_fn
from .util import image_to_level

def focal_loss(y_true, y_pred, alpha = .25, gamma = 2., weight = None, reduce = tf.reduce_mean):
    return focal_binary_cross_entropy(y_true, y_pred, alpha = alpha, gamma = gamma, weight = weight, reduce = reduce)

class YoloLoss(tf.keras.layers.Layer):
    def __init__(self, score_loss = binary_cross_entropy, class_loss = focal_loss, bbox_loss = ciou,
                 decode_bbox = True, valid_inside_anchor = False, weight = None,
                 assign = max_iou, sampler = None,
                 clip_ratio = 16 / 1000,
                 batch_size = 1,
                 missing_value = 0., dtype = tf.float32, **kwargs):
        kwargs["dtype"] = dtype
        super(YoloLoss, self).__init__(**kwargs)
        self.score_loss = score_loss
        self.class_loss = class_loss
        self.bbox_loss = bbox_loss
        self.decode_bbox = decode_bbox
        self.valid_inside_anchor = valid_inside_anchor
        self.weight = weight
        self.assign = assign
        self.sampler = sampler
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        self.missing_value = missing_value
        
        target = functools.partial(self.target, assign = self.assign, sampler = self.sampler, decode_bbox = self.decode_bbox)
        loss = functools.partial(self.loss, score_loss = self.score_loss, class_loss = self.class_loss, bbox_loss = self.bbox_loss, sampling = self.sampler is not None, weight = self.weight, missing_value = self.missing_value)
        target_eval = functools.partial(self.target, assign = self.assign, sampler = None, decode_bbox = self.decode_bbox)
        loss_eval = functools.partial(self.loss, score_loss = self.score_loss, class_loss = self.class_loss, bbox_loss = self.bbox_loss, sampling = False, weight = self.weight, missing_value = self.missing_value)
        self.target_func = tf.keras.layers.Lambda(lambda args: map_fn(target, *args, dtype = (tf.int8, self.dtype, self.dtype), batch_size = self.batch_size), dtype = self.dtype, name = "target")
        self.loss_func = tf.keras.layers.Lambda(lambda args: loss(*args), dtype = self.dtype, name = "loss")
        self.target_eval_func = tf.keras.layers.Lambda(lambda args: map_fn(target_eval, *args, dtype = (tf.int8, self.dtype, self.dtype), batch_size = self.batch_size), dtype = self.dtype, name = "target_eval")
        self.loss_eval_func = tf.keras.layers.Lambda(lambda args: loss_eval(*args), dtype = self.dtype, name = "loss_eval")
        
    @staticmethod
    @tf.function
    def target(y_true, bbox_true, score_pred, logit_pred, anchors, assign = max_iou, sampler = None, decode_bbox = True):
        """
        Args:
            y_true = label #(padded_num_true, 1 or num_class)
            bbox_true = [[x1, y1, x2, y2], ...] #(padded_num_true, 4)
            score_pred = classifier confidence score #(num_anchors, 1)
            logit_pred = classifier logit #(num_anchors, num_class)
            anchors = [[x1, y1, x2, y2], ...] #(num_anchors, 4)

        Returns:
            state = -1 : negative / 0 : neutral / 1 : positive #(num_anchors, 1)
            y_true = label #(num_anchors, 1 or num_class)
            bbox_true = [[x1, y1, x2, y2], ...] #(num_anchors, 4)
        """
        valid_indices = tf.where(tf.reduce_any(tf.greater(bbox_true, 0), axis = -1))[:, 0]
        bbox_true = tf.gather(bbox_true, valid_indices)
        if tf.keras.backend.int_shape(logit_pred)[-1] == 1:
            y_true = tf.ones([tf.shape(y_true)[0], 1], dtype = tf.float32)
        else:
            y_true = tf.gather(y_true, valid_indices)
            
        true_indices, positive_indices, negative_indices = assign(y_true, bbox_true, tf.multiply(logit_pred, score_pred), anchors)
        if sampler is not None:
            true_indices, positive_indices, negative_indices = sampler(true_indices, positive_indices, negative_indices)

        _positive_indices = tf.expand_dims(positive_indices, axis = -1)
        negative_indices = tf.expand_dims(negative_indices, axis = -1)

        pred_count = tf.shape(anchors)[0]
        state = tf.zeros([pred_count, 1], dtype = tf.int8)
        positive_state = tf.tensor_scatter_nd_update(state, _positive_indices, tf.ones_like(_positive_indices, dtype = tf.int8))
        state = tf.tensor_scatter_nd_update(positive_state, negative_indices, -tf.ones_like(negative_indices, dtype = tf.int8))

        n_class = tf.shape(y_true)[-1]
        _y_true = tf.cond(tf.equal(n_class, 1), true_fn = lambda: tf.zeros([pred_count, 1], dtype = tf.float32), false_fn = lambda: tf.pad(tf.ones([pred_count, 1], dtype = tf.float32), [[0, 0], [0, n_class - 1]]))
        y_true = tf.gather(y_true, true_indices)
        _y_true = tf.tensor_scatter_nd_update(_y_true, _positive_indices, tf.cast(y_true, tf.float32))
        _bbox_true = tf.zeros([pred_count, 4], dtype = tf.float32)
        if tf.keras.backend.int_shape(true_indices)[0] != 0:
            bbox_true = tf.gather(bbox_true, true_indices)
            if not decode_bbox:
                anchors = tf.gather(anchors, positive_indices)
                bbox_true = bbox2yolo(bbox_true, anchors)
            _bbox_true = tf.tensor_scatter_nd_update(_bbox_true, _positive_indices, bbox_true)
        return state, _y_true, _bbox_true
    
    @staticmethod
    @tf.function
    def loss(state_list, y_true_list, bbox_true_list, conf_pred_list, y_pred_list, bbox_pred_list, score_loss = binary_cross_entropy, class_loss = focal_loss, bbox_loss = ciou, sampling = False, weight = None, missing_value = 0.):
        """
        Args:
            state_list = level state list #[(N, num_anchors_1, 1), ..., (N, num_anchors_n, 1)]
            y_true_list = level label list #[(N, num_anchors_1, 1 or num_class), ..., (N, num_anchors_n, 1 or num_class)]
            bbox_true_list = level bbox_true list #[(N, num_anchors_1, 4), ..., (N, num_anchors_n, 4)]
            conf_pred_list = level score list #[(N, num_anchors_1, 1), ..., (N, num_anchors_n, 1)]
            y_pred_list = level logits list #[(N, num_anchors_1, num_class), ..., (N, num_anchors_n, num_class)]
            bbox_pred_list = level bbox_pred list #[(N, num_anchors_1, 4), ..., (N, num_anchors_n, 4)]

        Returns:
            score_loss_list = level confidence score loss #[(1,), ..., (1,)]
            class_loss_list = level class loss #[(1,), ..., (1,)]
            bbox_loss_list = level bbox loss #[(1,), ..., (1,)]
        """
        score_loss_list = []
        class_loss_list = []
        bbox_loss_list = []
        for i in range(len(state_list)):
            state, y_true, bbox_true = state_list[i], y_true_list[i], bbox_true_list[i]
            conf_pred, y_pred, bbox_pred = conf_pred_list[i], y_pred_list[i], bbox_pred_list[i]
            
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
            
            _score_loss = score_loss(positive_weight, conf_pred, reduce = None)
            _score_loss = weight_reduce_loss(_score_loss, weight = target_weight, avg_factor = avg_factor)
            
            _class_loss = class_loss(y_true, y_pred, weight = weight, reduce = None)
            _class_loss = weight_reduce_loss(_class_loss, weight = positive_weight, avg_factor = avg_factor)
            
            _bbox_loss = bbox_loss(bbox_true, bbox_pred, reduce = None)
            _bbox_loss = weight_reduce_loss(_bbox_loss, weight = positive_weight, avg_factor = avg_factor)
            
            _score_loss = tf.where(tf.logical_or(tf.math.is_nan(_score_loss), tf.math.is_inf(_score_loss)), tf.cast(missing_value, dtype), _score_loss)
            _class_loss = tf.where(tf.logical_or(tf.math.is_nan(_class_loss), tf.math.is_inf(_class_loss)), tf.cast(missing_value, dtype), _class_loss)
            _bbox_loss = tf.where(tf.logical_or(tf.math.is_nan(_bbox_loss), tf.math.is_inf(_bbox_loss)), tf.cast(missing_value, dtype), _bbox_loss)
            
            score_loss_list.append(_score_loss)
            class_loss_list.append(_class_loss)
            bbox_loss_list.append(_bbox_loss)
        if len(score_loss_list) == 1:
            score_loss_list = score_loss_list[0]
        if len(class_loss_list) == 1:
            class_loss_list = class_loss_list[0]
        if len(bbox_loss_list) == 1:
            bbox_loss_list = bbox_loss_list[0]
        return score_loss_list, class_loss_list, bbox_loss_list
    
    def call(self, inputs, outputs, training = None):
        y_true, bbox_true = inputs
        score_pred, logit_pred, bbox_pred, anchors = outputs
        if not isinstance(score_pred, (tuple, list)):
            score_pred = [score_pred]
        if not isinstance(logit_pred, (tuple, list)):
            logit_pred = [logit_pred]
        if not isinstance(bbox_pred, (tuple, list)):
            bbox_pred = [bbox_pred]
        if not isinstance(anchors, (tuple, list)):
            anchors = [anchors]
        if self.valid_inside_anchor:
            score_pred_list, logit_pred_list, bbox_pred_list, anchors_list = [], [], [], []
            for i, anchor in enumerate(anchors):
                valid_flags = tf.logical_and(tf.less_equal(anchor[..., 2], 1),
                                             tf.logical_and(tf.less_equal(anchor[..., 3], 1),
                                                            tf.logical_and(tf.greater_equal(anchor[..., 0], 0),
                                                                           tf.greater_equal(anchor[..., 1], 0))))
                #valid_indices = tf.range(tf.shape(anchor)[0])[valid_flags]
                valid_indices = tf.where(valid_flags)[:, 0]
                score_pred_list.append(tf.gather(score_pred[i], valid_indices, axis = 1))
                logit_pred_list.append(tf.gather(logit_pred[i], valid_indices, axis = 1))
                bbox_pred_list.append(tf.gather(bbox_pred[i], valid_indices, axis = 1))
                anchors_list.append(tf.gather(anchor, valid_indices))
        else:
            score_pred_list = score_pred
            logit_pred_list = logit_pred
            bbox_pred_list = bbox_pred
            anchors_list = anchors
            
        concat_score_pred = tf.concat(score_pred_list, axis = -2)
        concat_logit_pred = tf.concat(logit_pred_list, axis = -2)
        concat_anchors = tf.tile(tf.expand_dims(tf.concat(anchors_list, axis = 0), axis = 0), [tf.shape(bbox_true)[0], 1, 1])
        if training:
            state, y_true, bbox_true = self.target_func([y_true, bbox_true, concat_score_pred, concat_logit_pred, concat_anchors])
        else:
            state, y_true, bbox_true = self.target_eval_func([y_true, bbox_true, concat_score_pred, concat_logit_pred, concat_anchors])
        if self.decode_bbox:
            bbox_pred_list = [yolo2bbox(anchors_list[i], bbox_pred_list[i], clip_ratio = self.clip_ratio) for i in range(len(anchors_list))]
    
        n_level = [tf.shape(anchor)[0] for anchor in anchors_list]
        state_list = image_to_level(state, n_level)
        y_true_list = image_to_level(y_true, n_level)
        bbox_true_list = image_to_level(bbox_true, n_level)
        
        if training:
            score_loss, class_loss, bbox_loss = self.loss_func([state_list, y_true_list, bbox_true_list, score_pred_list, logit_pred_list, bbox_pred_list])
        else:
            score_loss, class_loss, bbox_loss = self.loss_eval_func([state_list, y_true_list, bbox_true_list, score_pred_list, logit_pred_list, bbox_pred_list])
        return score_loss, class_loss, bbox_loss