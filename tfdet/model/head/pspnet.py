import tensorflow as tf
import numpy as np

from .fcn import UpsamplingFeature

class PoolingPyramidModule(tf.keras.layers.Layer):
    def __init__(self, pool_scale = [1, 2, 3, 6], n_feature = 512, method = "bilinear", normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu, **kwargs):
        super(PoolingPyramidModule, self).__init__(**kwargs)
        self.pool_scale = pool_scale
        self.n_feature = n_feature
        self.normalize = normalize
        self.activation = activation
        self.method = method

    def build(self, input_shape):
        target_size = input_shape[1:3]
        self.pool_size = [np.divide(target_size, scale).astype(int) for scale in self.pool_scale]
        self.resize = tf.keras.layers.Lambda(lambda args, target_size, method: tf.image.resize(args, target_size, method = method), arguments = {"target_size":target_size, "method":self.method})
        self.layers = []
        for size in self.pool_size:
            layer = [tf.keras.layers.AveragePooling2D(size)]
            layer.append(tf.keras.layers.Conv2D(self.n_feature, 1, use_bias = self.normalize is None, kernel_initializer = "he_normal"))
            if self.normalize is not None:
                layer.append(self.normalize())
            layer.append(tf.keras.layers.Activation(self.activation))
            layer.append(self.resize)
            self.layers.append(layer)
        
    def call(self, inputs):
        out = []
        for layer in self.layers:
            o = inputs
            for l in layer:
                o = l(o)
            out.append(o)
        return out
    
    def get_config(self):
        config = super(PoolingPyramidModule, self).get_config()
        config["pool_scale"] = self.pool_scale
        config["n_feature"] = self.n_feature
        config["normalize"] = self.normalize
        config["activation"] = self.activation
        config["method"] = self.method
        return config

def pspnet(feature, n_class = 35, n_feature = 512, pool_scale = [1, 2, 3, 6], method = "bilinear", logits_activation = tf.keras.activations.sigmoid, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    #https://arxiv.org/abs/1612.01105
    if not isinstance(feature, list):
        feature = [feature]
    
    feature = UpsamplingFeature(concat = True, method = method, name = "upsampling_feature")(feature)
    out = PoolingPyramidModule(pool_scale, n_feature, method = method, normalize = normalize, activation = activation, name = "pooling_pyramoid_feature")(feature)
    out = tf.keras.layers.Concatenate(axis = -1, name = "feature_concat")([feature] + out)
    
    out = tf.keras.layers.Conv2D(n_feature, 3, padding = "same", use_bias = normalize is None, kernel_initializer = "he_normal", name = "feature_conv")(out)
    if normalize is not None:
        out = normalize(name = "feature_norm")(out)
    out = tf.keras.layers.Activation(activation, name = "feature_act")(out)
    
    out = tf.keras.layers.Conv2D(n_class, 1, use_bias = True, activation = logits_activation, kernel_initializer = "he_normal", name = "logits")(out)
    return out