import tensorflow as tf
import cv2
import numpy as np

from tfdet.core.util.anodet import feature_extract
from tfdet.core.util.distance import mahalanobis

class FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, sampling_index = None, memory_reduce = True, **kwargs):
        super(FeatureExtractor, self).__init__(**kwargs) 
        self.sampling_index = sampling_index
        self.memory_reduce = memory_reduce
        
    def call(self, inputs):
        out = feature_extract(inputs, sampling_index = self.sampling_index, pool_size = 1, sub_sampling = False, concat = True, memory_reduce = self.memory_reduce)
        return out
    
    def get_config(self):
        config = super(FeatureExtractor, self).get_config()
        config["sampling_index"] = self.sampling_index
        config["memory_reduce"] = self.memory_reduce
        return config
        
class Head(tf.keras.layers.Layer):
    def __init__(self, mean, cvar_inv = None, image_shape = [224, 224], sigma = 4, metric = mahalanobis, method = "bilinear", batch_size = 1, **kwargs):
        super(Head, self).__init__(**kwargs) 
        if isinstance(mean, tuple):
            mean, cvar_inv = mean
        self.mean = mean
        self.cvar_inv = cvar_inv
        self.image_shape = image_shape
        self.sigma = sigma
        self.metric = metric
        self.method = method
        self.batch_size = batch_size
        
        self.kernel = (2 * round(4 * sigma) + 1,) * 2
    
    def call(self, inputs):
        b = tf.shape(inputs)[0]
        h, w, c = tf.keras.backend.int_shape(inputs)[1:]
        feature = tf.reshape(inputs, [b, h * w, c])
        mask = tf.zeros((h * w, b, 1))
        mask = tf.while_loop(lambda index, mask: index < (h * w),
                             lambda index, mask: (index + 1, tf.tensor_scatter_nd_update(mask, tf.stack([tf.ones(b, dtype = tf.int32) * index, tf.range(b)], axis = -1), mahalanobis(feature[:, index], self.mean[index], self.cvar_inv[index]))),
                             (0, mask),
                             parallel_iterations = self.batch_size)[1]
        mask = tf.reshape(tf.transpose(mask, [1, 0, 2]), [b, h, w, 1])

        #upsampling
        mask = tf.image.resize(mask, self.image_shape, method = self.method)
        
        #gaussian smoothing
        if 0 < self.sigma:
            mask = tf.py_function(lambda *args: np.array([cv2.GaussianBlur(m, self.kernel, self.sigma) for m in args[0].numpy()]), inp = [mask], Tout = mask.dtype)
            mask = tf.reshape(mask, [-1, *self.image_shape, 1])
        return tf.expand_dims(tf.reduce_max(mask, axis = (1, 2, 3)), axis = -1), mask
    
    def get_config(self):
        config = super(Head, self).get_config()
        config["mean"] = self.mean
        config["cvar_inv"] = self.cvar_inv
        config["image_shape"] = self.image_shape
        config["sigma"] = self.sigma
        config["metric"] = self.metric
        config["method"] = self.method
        config["batch_size"] = self.batch_size
        return config
        