import functools

import tensorflow as tf

from tfdet.core.loss import categorical_cross_entropy, resize_loss

class FusedSemanticLoss(tf.keras.layers.Layer):
    def __init__(self, loss = categorical_cross_entropy,
                 weight = None, method = "bilinear", missing_value = 0., dtype = tf.float32, **kwargs):
        kwargs["dtype"] = dtype
        super(FusedSemanticLoss, self).__init__(**kwargs)
        self._loss = loss
        self.weight = weight
        self.method = method
        self.missing_value = missing_value
        
        _loss = functools.partial(self.loss, loss = self._loss, weight = self.weight, method = self.method, missing_value = self.missing_value)
        self.loss_func = tf.keras.layers.Lambda(lambda args: _loss(*args), dtype = self.dtype, name = "loss")
    
    @staticmethod
    @tf.function
    def loss(y_true, mask_true, semantic_pred, loss = categorical_cross_entropy, weight = None, method = "bilinear", missing_value = 0.):
        """
        Args:
            y_true = #(N, padded_num_true, 1 or num_class)
            mask_true = #(N, padded_num_true, h, w, 1)
            semantic_pred = #(N, h, w, n_class)

        Returns:
            semantic_loss = #(1,)
        """
        mask_shape = tf.shape(mask_true)
        semantic_shape = tf.shape(semantic_pred)

        label_true = tf.cond(tf.equal(tf.shape(y_true)[2], 1), true_fn = lambda: tf.cast(y_true[..., 0], tf.int32), false_fn = lambda: tf.argmax(y_true, axis = -1, output_type = tf.int32))

        semantic_true = tf.reshape(mask_true, [-1, mask_shape[2], mask_shape[3], 1])
        semantic_true = tf.image.resize(semantic_true, semantic_shape[-3:-1], method = method)
        semantic_true = tf.reshape(semantic_true, [mask_shape[0], mask_shape[1], semantic_shape[-3], semantic_shape[-2]])
        semantic_true = tf.clip_by_value(tf.round(semantic_true), 0., 1.)
        semantic_true = tf.multiply(tf.expand_dims(tf.expand_dims(label_true, axis = -1), axis = -1), tf.cast(semantic_true, tf.int32))
        semantic_true = tf.reduce_max(semantic_true, axis = 1)
        semantic_true = tf.one_hot(semantic_true, semantic_shape[-1])
        
        _loss = loss(semantic_true, semantic_pred, weight = weight)
        _loss = tf.where(tf.logical_or(tf.math.is_nan(_loss), tf.math.is_inf(_loss)), tf.cast(missing_value, _loss.dtype), _loss)
        return _loss
        
    def call(self, inputs, outputs, bbox = True, mask = True):
        y_true, mask_true = inputs[:2]
        if 2 < len(inputs) and tf.keras.backend.int_shape(mask_true)[-1] == 4:
            mask_true = inputs[2]
        semantic_pred = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
     
        semantic_loss = self.loss_func([y_true, mask_true, semantic_pred])
        return semantic_loss
    
class ResizeMaskLoss(tf.keras.layers.Layer):
    def __init__(self, loss = categorical_cross_entropy,
                 weight = None, method = "bilinear", missing_value = 0., dtype = tf.float32, **kwargs):
        kwargs["dtype"] = dtype
        super(ResizeMaskLoss, self).__init__(**kwargs)
        self._loss = loss
        self.weight = weight
        self.method = method
        self.missing_value = missing_value
        
        _loss = functools.partial(self.loss, loss = self._loss, weight = self.weight, method = self.method, missing_value = self.missing_value)
        self.loss_func = tf.keras.layers.Lambda(lambda args: _loss(*args), dtype = self.dtype, name = "loss")
    
    @staticmethod
    @tf.function
    def loss(mask_true, mask_pred, loss = categorical_cross_entropy, weight = None, method = "bilinear", missing_value = 0.):
        """
        Args:
            mask_true = #(N, h, w, 1 or n_class)
            mask_pred = #(N, h, w, n_class)

        Returns:
            semantic_loss = #(1,)
        """
        _loss = resize_loss(mask_true, mask_pred, loss = loss, method = method, weight = weight, reduce = None)
        _loss = tf.reduce_mean(_loss)
        _loss = tf.where(tf.logical_or(tf.math.is_nan(_loss), tf.math.is_inf(_loss)), tf.cast(missing_value, _loss.dtype), _loss)
        return _loss
        
    def call(self, inputs, outputs):
        mask_true = inputs[-1] if isinstance(inputs, (tuple, list)) else inputs
        mask_pred = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
        loss = self.loss_func([mask_true, mask_pred])
        return loss