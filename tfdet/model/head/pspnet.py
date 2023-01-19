import tensorflow as tf
import numpy as np

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

class PoolingPyramidModule(tf.keras.layers.Layer):
    def __init__(self, pool_scale = [1, 2, 3, 6], n_feature = 512, method = "bilinear", convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu, **kwargs):
        super(PoolingPyramidModule, self).__init__(**kwargs)
        self.pool_scale = pool_scale
        self.n_feature = n_feature
        self.method = method
        self.convolution = convolution
        self.normalize = normalize
        self.activation = activation

    def build(self, input_shape):
        target_size = input_shape[1:3]
        self.pool_size = [np.divide(target_size, scale).astype(int) for scale in self.pool_scale]
        self.resize = tf.keras.layers.Lambda(lambda args, target_size, method: tf.image.resize(args, target_size, method = method), arguments = {"target_size":target_size, "method":self.method})
        self.layers = []
        for size in self.pool_size:
            layer = [tf.keras.layers.AveragePooling2D(size)]
            layer.append(self.convolution(self.n_feature, 1, use_bias = self.normalize is None))
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
        config["method"] = self.method
        return config

def pspnet_head(feature, n_class = 35, n_feature = 512, pool_scale = [1, 2, 3, 6], method = "bilinear", logits_activation = tf.keras.activations.sigmoid, convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    #https://arxiv.org/abs/1612.01105
    if isinstance(feature, list):
        feature = feature[-1]
    out = PoolingPyramidModule(pool_scale, n_feature, method = method, convolution = convolution, normalize = normalize, activation = activation, name = "pooling_pyramoid_feature")(feature)
    out = tf.keras.layers.Concatenate(axis = -1, name = "feature_concat")([feature] + out)
    
    out = convolution(n_feature, 3, padding = "same", use_bias = normalize is None, name = "feature_conv")(out)
    if normalize is not None:
        out = normalize(name = "feature_norm")(out)
    out = tf.keras.layers.Activation(activation, name = "feature_act")(out)
    
    out = convolution(n_class, 1, use_bias = True, name = "logits")(out)
    out = tf.keras.layers.Activation(logits_activation if logits_activation is not None else tf.keras.activations.linear, dtype = tf.float32, name = "logits_act")(out)
    return out