import tensorflow as tf

from ..head import unet_head, unet_2plus_head

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def unet(feature, n_class = 35, n_feature = 64, n_depth = 5, method = "bilinear", neck = None, logits_activation = tf.keras.activations.sigmoid, convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    if neck is not None:
        feature = neck(name = "neck")(feature)
    out = unet_head(feature, n_class = n_class, n_feature = n_feature, n_depth = n_depth, method = method, logits_activation = logits_activation, convolution = convolution, normalize = normalize, activation = activation)
    return out

def unet_2plus(feature, n_class = 35, n_feature = 256, deep_supervision = False, method = "bilinear", neck = None, logits_activation = tf.keras.activations.sigmoid, convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    if neck is not None:
        feature = neck(name = "neck")(feature)
    out = unet_2plus_head(feature, n_class = n_class, n_feature = n_feature, deep_supervision = deep_supervision, method = method, logits_activation = logits_activation, convolution = convolution, normalize = normalize, activation = activation)
    return out