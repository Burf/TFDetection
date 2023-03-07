import tensorflow as tf
import cv2
import numpy as np

from tfdet.core.ops import feature_extract, euclidean_matrix

class FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, sampling_index = None, pool_size = 3, memory_reduce = False, **kwargs):
        super(FeatureExtractor, self).__init__(**kwargs) 
        self.sampling_index = sampling_index
        self.pool_size = pool_size
        self.memory_reduce = memory_reduce
        
    def call(self, inputs):
        out = feature_extract(inputs, sampling_index = self.sampling_index, pool_size = self.pool_size, sub_sampling = False, concat = True, memory_reduce = self.memory_reduce)
        return out
    
    def get_config(self):
        config = super(FeatureExtractor, self).get_config()
        config["sampling_index"] = self.sampling_index
        config["pool_size"] = self.pool_size
        config["memory_reduce"] = self.memory_reduce
        return config
        
class Head(tf.keras.layers.Layer):
    def __init__(self, feature_vector, image_shape = [224, 224], k = 9, sigma = 4, method = "bilinear", **kwargs):
        super(Head, self).__init__(**kwargs)
        self.feature_vector = feature_vector
        self.image_shape = image_shape
        self.k = k
        self.sigma = sigma
        self.method = method
        
        self.kernel = (2 * round(4 * sigma) + 1,) * 2
    
    def call(self, inputs):
        b = tf.shape(inputs)[0]
        h, w, c = tf.keras.backend.int_shape(inputs)[1:]
        feature = tf.reshape(inputs, [b * h * w, c])
        score = tf.sort(euclidean_matrix(feature, self.feature_vector), axis = -1)[..., :self.k]
        mask = tf.reshape(score[..., 0], [b, h, w, 1])
        score = tf.reshape(score, [b, h * w, -1])
        #conf = tf.gather_nd(score, tf.stack([tf.range(b), tf.cast(tf.argmax(score[..., 0], axis = -1), tf.int32)], axis = -1))
        conf = tf.gather(score, tf.argmax(score[..., 0], axis = -1), batch_dims = 1)
        exp_conf = tf.exp(conf)
        weight = 1 - tf.reduce_max(exp_conf, axis = -1) / tf.reduce_sum(exp_conf, axis = -1)
        score = tf.reduce_max(score[..., 0], axis = -1) * weight
        score = tf.expand_dims(score, axis = -1)

        #upsampling
        mask = tf.image.resize(mask, self.image_shape, method = self.method)
        
        #gaussian smoothing
        if 0 < self.sigma:
            mask = tf.py_function(lambda *args: np.array([cv2.GaussianBlur(m, self.kernel, self.sigma) for m in args[0].numpy()]), inp = [mask], Tout = mask.dtype)
            mask = tf.reshape(mask, [-1, *self.image_shape, 1])
        return score, mask
    
    def get_config(self):
        config = super(Head, self).get_config()
        config["k"] = self.k
        config["image_shape"] = self.image_shape
        config["sigma"] = self.sigma
        config["method"] = self.method
        return config
        
def patch_core_head(feature, feature_vector = None, image_shape = [224, 224], k = 9, sampling_index = None, pool_size = 3, sigma = 4, method = "bilinear", memory_reduce = False):
    feature = FeatureExtractor(sampling_index = sampling_index, pool_size = pool_size, memory_reduce = memory_reduce, name = "feature_extractor")(feature)
    if feature_vector is not None:
        score, mask = Head(feature_vector = feature_vector, image_shape = image_shape, k = k, sigma = sigma, method = method, name = "patch_core")(feature)
        return score, mask
    else:
        return feature