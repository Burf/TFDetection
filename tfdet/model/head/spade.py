import tensorflow as tf
import cv2
import numpy as np

from tfdet.core.util import feature_extract, euclidean, euclidean_matrix

class FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, sampling_index = None, **kwargs):
        super(FeatureExtractor, self).__init__(**kwargs) 
        self.sampling_index = sampling_index
        
    def call(self, inputs):
        out = feature_extract(inputs, sampling_index = self.sampling_index, pool_size = 1, sub_sampling = True, concat = False)
        return out
    
    def get_config(self):
        config = super(FeatureExtractor, self).get_config()
        config["sampling_index"] = self.sampling_index
        return config
        
class Head(tf.keras.layers.Layer):
    def __init__(self, feature_vector, image_shape = [224, 224], k = 50, sigma = 4, method = "bilinear", **kwargs):
        super(Head, self).__init__(**kwargs)
        self.feature_vector = feature_vector
        self.image_shape = image_shape
        self.k = k
        self.sigma = sigma
        self.method = method
        
        self.kernel = (2 * round(4 * sigma) + 1,) * 2
    
    def call(self, inputs):
        #calculate image score with extract top-k feature by euclidean distance for gap feature
        dist = euclidean_matrix(tf.squeeze(inputs[-1], axis = [1, 2]), tf.squeeze(self.feature_vector[-1], axis = [1, 2]))
        score = tf.sort(dist, axis = -1)[..., :self.k]
        score = tf.reduce_mean(score, axis = -1, keepdims = True)
        indices = tf.argsort(dist, axis = -1)[..., :self.k]

        #calculate pixel score with extract top-k feature by euclidean distance for feature
        mask = []
        for tr_feature, pred_feature in zip(self.feature_vector[:-1], inputs[:-1]):
            tr_feature = tf.gather(tr_feature, indices)
            pred_feature = tf.expand_dims(pred_feature, axis = 1)
            dist = euclidean(tr_feature, pred_feature)
            m = tf.reduce_min(dist, axis = 1)
            m = tf.image.resize(tf.expand_dims(m, axis = -1), self.image_shape, method = self.method) #upsampling
            mask.append(m)
        mask = tf.reduce_mean(mask, axis = 0)
        
        #gaussian smoothing
        if 0 < self.sigma:
            mask = tf.py_function(lambda *args: np.array([cv2.GaussianBlur(m, self.kernel, self.sigma) for m in args[0].numpy()]), inp = [mask], Tout = mask.dtype)
            mask = tf.reshape(mask, [-1, *self.image_shape, 1])
        return score, mask
    
    def get_config(self):
        config = super(Head, self).get_config()
        config["feature_vector"] = self.feature_vector
        config["k"] = self.k
        config["image_shape"] = self.image_shape
        config["sigma"] = self.sigma
        config["method"] = self.method
        return config
        