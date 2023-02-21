import tensorflow as tf

from tfdet.core.ops import AdaptiveAveragePooling2D, AdaptiveMaxPooling2D

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def head_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def normalize(axis = -1, momentum = 0.9, epsilon = 1e-5, **kwargs):
    return tf.keras.layers.BatchNormalization(axis = axis, momentum = momentum, epsilon = epsilon, **kwargs)

class PoolingPyramidModule(tf.keras.layers.Layer):
    def __init__(self, pool_scale = [1, 2, 3, 6], n_feature = 512, max_pooling = False, method = "bilinear",
                 convolution = conv, normalize = normalize, activation = tf.keras.activations.relu, **kwargs):
        super(PoolingPyramidModule, self).__init__(**kwargs)
        self.pool_scale = [pool_scale] if not isinstance(pool_scale, (tuple, list)) else pool_scale
        self.n_feature = n_feature
        self.max_pooling = max_pooling
        self.method = method
        self.convolution = convolution
        self.normalize = normalize
        self.activation = activation

        self.layers = []
        for scale in self.pool_scale:
            layer = [(AdaptiveMaxPooling2D if max_pooling else AdaptiveAveragePooling2D)(scale)]
            layer.append(self.convolution(self.n_feature, 1, use_bias = self.normalize is None))
            if self.normalize is not None:
                if scale == 1: #for bn moving_varriance nan bug fix
                    layer.append(tf.keras.layers.Reshape([self.n_feature]))
                    layer.append(self.normalize())
                    layer.append(tf.keras.layers.Reshape([1, 1, self.n_feature]))
                else:
                    layer.append(self.normalize())
            layer.append(tf.keras.layers.Activation(self.activation))
            self.layers.append(layer)
        self.upsample = tf.keras.layers.Lambda(lambda args: tf.image.resize(args[0], args[1], method = method), name = "upsample")
        
    def call(self, inputs):
        target_size = tf.shape(inputs)[1:3]
        out = []
        for layer in self.layers:
            o = inputs
            for l in layer:
                o = l(o)
            o = self.upsample([o, target_size])
            out.append(o)
        return out
    
    def get_config(self):
        config = super(PoolingPyramidModule, self).get_config()
        config["pool_scale"] = self.pool_scale
        config["n_feature"] = self.n_feature
        config["max_pooling"] = self.max_pooling
        config["method"] = self.method
        return config

def pspnet_head(feature, n_class = 35, n_feature = 512, pool_scale = [1, 2, 3, 6], max_pooling = False, method = "bilinear", dropout_rate = 0.1,
                logits_convolution = head_conv, logits_activation = tf.keras.activations.softmax,
                convolution = conv, normalize = normalize, activation = tf.keras.activations.relu):
    #https://arxiv.org/abs/1612.01105
    if isinstance(feature, (tuple, list)):
        feature = feature[-1]
    out = PoolingPyramidModule(pool_scale, n_feature, max_pooling = max_pooling, method = method, convolution = convolution, normalize = normalize, activation = activation, name = "pooling_pyramoid_feature")(feature)
    out = tf.keras.layers.Concatenate(axis = -1, name = "feature_concat")([feature] + out)
    
    out = convolution(n_feature, 3, padding = "same", use_bias = normalize is None, name = "feature_conv")(out)
    if normalize is not None:
        out = normalize(name = "feature_norm")(out)
    out = tf.keras.layers.Activation(activation, name = "feature_act")(out)
    
    if 0 < dropout_rate:
        out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = logits_convolution(n_class, 1, use_bias = True, name = "logits")(out)
    out = tf.keras.layers.Activation(logits_activation if logits_activation is not None else tf.keras.activations.linear, dtype = tf.float32, name = "logits_act")(out)
    return out