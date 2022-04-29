import tensorflow as tf

from ..head.fcn import UpsamplingFeature
from ..head.pspnet import PoolingPyramidModule

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def pspnet(feature, n_class = 35, n_feature = 512, pool_scale = [1, 2, 3, 6], method = "bilinear", logits_activation = tf.keras.activations.sigmoid, convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    #https://arxiv.org/abs/1612.01105
    if not isinstance(feature, list):
        feature = [feature]
    
    feature = UpsamplingFeature(concat = True, method = method, name = "upsampling_feature")(feature)
    out = PoolingPyramidModule(pool_scale, n_feature, method = method, convolution = convolution, normalize = normalize, activation = activation, name = "pooling_pyramoid_feature")(feature)
    out = tf.keras.layers.Concatenate(axis = -1, name = "feature_concat")([feature] + out)
    
    out = convolution(n_feature, 3, padding = "same", use_bias = normalize is None, name = "feature_conv")(out)
    if normalize is not None:
        out = normalize(name = "feature_norm")(out)
    out = tf.keras.layers.Activation(activation, name = "feature_act")(out)
    
    out = convolution(n_class, 1, use_bias = True, activation = logits_activation, name = "logits")(out)
    return out