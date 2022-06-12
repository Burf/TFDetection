import tensorflow as tf

from ..head import deeplab_v3_head

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def deeplab_v3(feature, n_class = 35, rate = [6, 12, 18], n_feature = 256, method = "bilinear", neck = None, logits_activation = tf.keras.activations.sigmoid, convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    if neck is not None:
        feature = neck(name = "neck")(feature)
    if isinstance(feature, list):
        feature = feature[-1]
    out = deeplab_v3_head(feature, n_class, rate, n_feature, method = method, logits_activation = logits_activation, convolution = convolution, normalize = normalize, activation = activation)
    return out

def deeplab_v3_plus(feature, n_class = 35, rate = [6, 12, 18], n_feature = 256, n_low_feature = 48, method = "bilinear", neck = None, logits_activation = tf.keras.activations.sigmoid, convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    if neck is not None:
        feature = neck(name = "neck")(feature)
    if not isinstance(feature, list):
        feature = [feature]
    out = deeplab_v3_head(feature, n_class, rate, n_feature, n_low_feature, method = method, logits_activation = logits_activation, convolution = convolution, normalize = normalize, activation = activation)
    return out