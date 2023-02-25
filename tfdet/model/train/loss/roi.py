import functools

import tensorflow as tf

from tfdet.core.assign import max_iou, random_sampler
from tfdet.core.bbox import bbox2delta, delta2bbox
from tfdet.core.loss import binary_cross_entropy, focal_categorical_cross_entropy, smooth_l1, weight_reduce_loss
from tfdet.core.util import map_fn

def smooth_l1_sigma1(y_true, y_pred, sigma = 1, reduce = tf.reduce_mean):
    return smooth_l1(y_true, y_pred, sigma = sigma, reduce = reduce)

def roi_assign(y_true, bbox_true, y_pred, bbox_pred, positive_threshold = 0.5, negative_threshold = 0.5, min_threshold = 0.5, match_low_quality = False, mode = "normal"):
    return max_iou(y_true, bbox_true, y_pred, bbox_pred, positive_threshold = positive_threshold, negative_threshold = negative_threshold, min_threshold = min_threshold, match_low_quality = match_low_quality, mode = mode)

def roi_sampler(true_indices, positive_indices, negative_indices, sampling_count = 512, positive_ratio = 0.25, return_count = False):
    return random_sampler(true_indices, positive_indices, negative_indices, sampling_count = sampling_count, positive_ratio = positive_ratio, return_count = return_count)

class RoiTarget(tf.keras.layers.Layer):
    def __init__(self, assign = roi_assign, sampler = roi_sampler,
                 mask_size = 28, method = "bilinear",
                 add_gt_in_sampler = True,
                 batch_size = 1, dtype = tf.float32, **kwargs):
        kwargs["dtype"] = dtype
        super(RoiTarget, self).__init__(**kwargs)
        self.assign = assign
        self.sampler = sampler
        self.mask_size = mask_size
        self.method = method
        self.add_gt_in_sampler = add_gt_in_sampler
        self.batch_size = batch_size
        
        target = functools.partial(self.target, assign = self.assign, sampler = self.sampler, mask_size = self.mask_size, method = self.method, add_gt_in_sampler = self.add_gt_in_sampler)
        target_eval = functools.partial(self.target, assign = self.assign, sampler = None, mask_size = self.mask_size, method = self.method)
        dtype = (tf.int8, self.dtype, self.dtype, self.dtype)
        mask_dtype = (tf.int8, self.dtype, self.dtype, tf.uint8, self.dtype)
        self.target_func = tf.keras.layers.Lambda(lambda args: map_fn(target, *args, dtype = (mask_dtype if 3 < len(args) else dtype), batch_size = batch_size), dtype = self.dtype, name = "target")
        self.target_eval_func = tf.keras.layers.Lambda(lambda args: map_fn(target_eval, *args, dtype = (mask_dtype if 3 < len(args) else dtype), batch_size = batch_size), dtype = self.dtype, name = "target_eval")
        
    @staticmethod
    @tf.function
    def target(y_true, bbox_true, proposals, mask_true = None, assign = roi_assign, sampler = roi_sampler, mask_size = 28,  method = "bilinear", add_gt_in_sampler = True):
        """
        Args:
            y_true = label #(padded_num_true, 1 or num_class)
            bbox_true = [[x1, y1, x2, y2], ...] #(padded_num_true, 4)
            proposals = [[x1, y1, x2, y2], ...] #(num_proposals, 4)
            mask_true = instance mask true(optional) #(padded_num_true, h, w, 1)

        Returns:
            state = -1 : negative / 0 : neutral / 1 : positive #(num_proposals, 1)
            y_true = label #(num_proposals, 1 or num_class)
            bbox_true = [[x1, y1, x2, y2], ...] #(num_proposals, 4)
            mask_true = mask(optional) #(num_proposals, mask_size, mask_size, 1)
        """
        valid_indices = tf.where(tf.reduce_any(tf.greater(bbox_true, 0), axis = -1))[:, 0]
        y_true = tf.gather(y_true, valid_indices)
        bbox_true = tf.gather(bbox_true, valid_indices)

        pred_count = tf.shape(proposals)[0]
        
        valid_indices = tf.where(tf.reduce_any(tf.greater(proposals, 0), axis = -1))[:, 0]
        proposals = tf.gather(proposals, valid_indices)
        true_indices, positive_indices, negative_indices = assign(tf.ones([tf.shape(bbox_true)[0], 1], dtype = tf.float32), bbox_true, tf.ones([tf.shape(proposals)[0], 1], dtype = tf.float32), proposals)
        if sampler is not None:
            if add_gt_in_sampler:
                bbox_indices = tf.range(tf.shape(bbox_true)[0])
                true_indices = tf.concat([true_indices, tf.cast(bbox_indices, true_indices.dtype)], axis = 0)
                positive_indices = tf.concat([positive_indices, tf.cast(bbox_indices + tf.shape(proposals)[0], positive_indices.dtype)], axis = 0)
                proposals = tf.concat([proposals, tf.cast(bbox_true, proposals.dtype)], axis = 0)
            true_indices, positive_indices, negative_indices, pred_count = sampler(true_indices, positive_indices, negative_indices, return_count = True)

        n_class = tf.shape(y_true)[-1]
        pred_indices = tf.concat([positive_indices, negative_indices], axis = 0)
        state = tf.expand_dims(tf.concat([tf.ones_like(positive_indices, dtype = tf.int8), -tf.ones_like(negative_indices, dtype = tf.int8)], axis = 0), axis = -1)
        y_true = tf.gather(y_true, true_indices)
        bbox_true = tf.gather(bbox_true, true_indices)
        _proposals = tf.gather(proposals, pred_indices)
        if tf.keras.backend.int_shape(positive_indices)[0] != 0:
            if mask_true is not None:
                proposals = tf.gather(proposals, positive_indices)
                mask_true = tf.gather(mask_true, true_indices)
                x1, y1, x2, y2 = tf.split(proposals, 4, axis = -1)
                mask_bbox = tf.concat([y1, x1, y2, x2], axis = -1)
                mask_true = tf.image.crop_and_resize(image = mask_true, boxes = tf.cast(mask_bbox, tf.float32), box_indices = tf.range(tf.shape(positive_indices)[0]), crop_size = [mask_size, mask_size], method = method)
                mask_true = tf.clip_by_value(tf.round(mask_true), 0., 1.)
                mask_true = tf.cast(mask_true, tf.uint8)
        else:
            mask_true = tf.zeros([tf.shape(positive_indices)[0], mask_size, mask_size, 1], dtype = tf.uint8)
        
        negative_count = tf.shape(negative_indices)[0]
        pad_count = tf.maximum(pred_count - tf.shape(pred_indices)[0], 0)
        state = tf.pad(state, [[0, pad_count], [0, 0]])
        y_true = tf.cond(tf.equal(n_class, 1), true_fn = lambda: tf.pad(y_true, [[0, negative_count + pad_count], [0, 0]]), false_fn = lambda: tf.concat([y_true, tf.cast(tf.pad(tf.ones([negative_count + pad_count, 1]), [[0, 0], [0, n_class - 1]]), y_true.dtype)], axis = 0))
        bbox_true = tf.pad(bbox_true, [[0, negative_count + pad_count], [0, 0]])
        _proposals = tf.pad(_proposals, [[0, pad_count], [0, 0]])
        if mask_true is not None:
            mask_true = tf.pad(mask_true, [[0, negative_count + pad_count], [0, 0], [0, 0], [0, 0]])
            return state, y_true, bbox_true, mask_true, _proposals
        else:
            return state, y_true, bbox_true, _proposals
    
    def call(self, inputs, outputs, training = None):
        if not isinstance(outputs, (tuple, list)):
            outputs = [outputs]
        
        y_true, bbox_true = inputs[:2]
        mask_true = None
        if 2 < len(inputs):
            mask_true = inputs[2]
        
        proposals = outputs[0]
        if not training and 1 < len(outputs):
            proposals = outputs[1]
        
        args = [arg for arg in [y_true, bbox_true, proposals, mask_true] if arg is not None]
        if training:
            out = self.target_func(args)
        else:
            out = self.target_eval_func(args)
        state, y_true, bbox_true = out[:3]
        if mask_true is not None:
            mask_true, proposals = out[3:]
        else:
            proposals = out[3]

        if mask_true is not None:
            return state, y_true, bbox_true, mask_true, proposals
        else:
            return state, y_true, bbox_true, proposals

class RoiBboxLoss(tf.keras.layers.Layer):
    def __init__(self, class_loss = focal_categorical_cross_entropy, bbox_loss = smooth_l1_sigma1,
                 sampling = True, decode_bbox = False, weight = None, background = True,
                 mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000,
                 missing_value = 0., dtype = tf.float32, **kwargs):
        kwargs["dtype"] = dtype
        super(RoiBboxLoss, self).__init__(**kwargs)
        self.class_loss = class_loss
        self.bbox_loss = bbox_loss
        self.sampling = sampling
        self.decode_bbox = decode_bbox
        self.weight = weight
        self.background = background
        self.mean = mean
        self.std = std
        self.clip_ratio = clip_ratio
        self.missing_value = missing_value
        
        loss = functools.partial(self.loss, class_loss = self.class_loss, bbox_loss = self.bbox_loss, sampling = self.sampling, weight = self.weight, background = self.background, decode_bbox = self.decode_bbox, mean = self.mean, std = self.std, clip_ratio = self.clip_ratio, missing_value = self.missing_value)
        loss_eval = functools.partial(self.loss, class_loss = self.class_loss, bbox_loss = self.bbox_loss, sampling = False, weight = self.weight, background = self.background, decode_bbox = self.decode_bbox, mean = self.mean, std = self.std, clip_ratio = self.clip_ratio, missing_value = self.missing_value)
        self.loss_func = tf.keras.layers.Lambda(lambda args: loss(*args), dtype = self.dtype, name = "loss")
        self.loss_eval_func = tf.keras.layers.Lambda(lambda args: loss_eval(*args), dtype = self.dtype, name = "loss_eval")
    
    @staticmethod
    @tf.function
    def loss(state, y_true, bbox_true, y_pred, bbox_pred, proposals, class_loss = focal_categorical_cross_entropy, bbox_loss = smooth_l1_sigma1, sampling = True, weight = None, background = True, decode_bbox = False, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, missing_value = 0.):
        """
        Args:
            state = #(N, num_proposals, 1)
            y_true = #(N, num_proposals, 1 or num_class)
            bbox_true = #(N, num_proposals, 4)
            y_pred = #(N, num_proposals, num_class)
            bbox_pred = #(N, num_proposals, num_class, 4)
            proposals = #(N, num_proposals, 4)

        Returns:
            class_loss = #(1,)
            bbox_loss = #(1,)
        """
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
        label_true = tf.argmax(y_true, axis = -1, output_type = tf.int32)
        if not background:
            y_true = y_true * positive_weight
        
        bbox_pred = tf.gather(bbox_pred, label_true, batch_dims = 2)
        if decode_bbox:
            bbox_pred = delta2bbox(proposals, bbox_pred, mean = mean, std = std, clip_ratio = clip_ratio)
        else:
            bbox_true = bbox2delta(bbox_true, proposals, mean = mean, std = std)

        _class_loss = class_loss(y_true, y_pred, weight = weight, reduce = None)
        _class_loss = weight_reduce_loss(_class_loss, weight = target_weight, avg_factor = avg_factor)
        
        _bbox_loss = bbox_loss(bbox_true, bbox_pred, reduce = None)
        _bbox_loss = weight_reduce_loss(_bbox_loss, weight = positive_weight, avg_factor = avg_factor)
        
        _class_loss = tf.where(tf.logical_or(tf.math.is_nan(_class_loss), tf.math.is_inf(_class_loss)), tf.cast(missing_value, dtype), _class_loss)
        _bbox_loss = tf.where(tf.logical_or(tf.math.is_nan(_bbox_loss), tf.math.is_inf(_bbox_loss)), tf.cast(missing_value, dtype), _bbox_loss)
        return _class_loss, _bbox_loss
    
    def call(self, inputs, outputs, training = None):
        state, y_true, bbox_true = inputs[:3]
        y_pred, bbox_pred, proposals = outputs[:3]
        
        if training:
            class_loss, bbox_loss = self.loss_func([state, y_true, bbox_true, y_pred, bbox_pred, proposals])
        else:
            class_loss, bbox_loss = self.loss_eval_func([state, y_true, bbox_true, y_pred, bbox_pred, proposals])
        return class_loss, bbox_loss

class RoiMaskLoss(tf.keras.layers.Layer):
    def __init__(self, loss = binary_cross_entropy,
                 missing_value = 0., dtype = tf.float32, **kwargs):
        kwargs["dtype"] = dtype
        super(RoiMaskLoss, self).__init__(**kwargs)
        self._loss = loss
        self.missing_value = missing_value
        
        _loss = functools.partial(self.loss, loss = self._loss, missing_value = self.missing_value)
        self.loss_func = tf.keras.layers.Lambda(lambda args: _loss(*args), dtype = self.dtype, name = "loss")
    
    @staticmethod
    @tf.function
    def loss(state, y_true, mask_true, mask_pred, loss = binary_cross_entropy, missing_value = 0.):
        """
        Args:
            state = #(N, num_proposals, 1)
            y_true = #(N, num_proposals, 1 or num_class)
            mask_true = #(N, num_proposals, mask_size, mask_size, 1)
            mask_pred = #(N, num_proposals, mask_size, mask_size, num_class)

        Returns:
            mask_loss = #(1,)
        """
        dtype = mask_pred.dtype
        n_true_class = tf.shape(y_true)[-1]
        n_pred_class = tf.shape(mask_pred)[-1]
        
        label_true = tf.cond(tf.equal(n_true_class, 1), true_fn = lambda: tf.cast(y_true[..., 0], tf.int32), false_fn = lambda: tf.argmax(y_true, axis = -1, output_type = tf.int32))
        
        mask_pred = tf.transpose(mask_pred, [0, 1, 4, 2, 3])
        mask_pred = tf.gather(mask_pred, label_true, batch_dims = 2)
        mask_pred = tf.expand_dims(mask_pred, axis = -1)
        
        positive_weight = tf.cast(tf.equal(state, 1), dtype)
        avg_factor = tf.reduce_sum(tf.maximum(tf.reduce_sum(positive_weight, axis = -2), tf.cast(1., dtype)))
        
        _mask_loss = loss(tf.cast(mask_true, tf.float32), mask_pred, reduce = None)
        _mask_loss = tf.reduce_mean(_mask_loss, axis = [-1, -2, -3])
        _mask_loss = weight_reduce_loss(_mask_loss, weight = positive_weight[..., 0], avg_factor = avg_factor)
        
        _mask_loss = tf.where(tf.logical_or(tf.math.is_nan(_mask_loss), tf.math.is_inf(_mask_loss)), tf.cast(missing_value, dtype), _mask_loss)
        return _mask_loss
    
    def call(self, inputs, outputs):
        state, y_true, bbox_true = inputs[:3]
        mask_true = None
        if 3 < len(inputs):
            mask_true = inputs[3]
        elif tf.keras.backend.int_shape(bbox_true)[-1] == 1:
            mask_true = bbox_true
            bbox_true = None
        mask_pred = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs

        mask_loss = self.loss_func([state, y_true, mask_true, mask_pred])
        return mask_loss