import tensorflow as tf

from ..head import pspnet_head
from ..neck import FeatureUpsample

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def pspnet(feature, n_class = 35, n_feature = 512, pool_scale = [1, 2, 3, 6], method = "bilinear", neck = FeatureUpsample, logits_activation = tf.keras.activations.sigmoid, convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    if neck is not None:
        feature = neck(name = "neck")(feature)
    out = pspnet_head(feature, n_class = n_class, n_feature = n_feature, pool_scale = pool_scale, method = method, logits_activation = logits_activation, convolution = convolution, normalize = normalize, activation = activation)
    return out