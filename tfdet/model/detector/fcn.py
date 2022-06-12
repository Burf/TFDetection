import tensorflow as tf

from ..head import fcn_head
from ..neck import FeatureUpsample

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def fcn(feature, n_class = 35, n_feature = 512, n_depth = 2, neck = FeatureUpsample, logits_activation = tf.keras.activations.sigmoid, convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    if neck is not None:
        feature = neck(name = "neck")(feature)
    out = fcn_head(feature, n_class = n_class, n_feature = n_feature, n_depth = n_depth, logits_activation = logits_activation, convolution = convolution, normalize = normalize, activation = activation)
    return out