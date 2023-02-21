import tensorflow as tf

from ..head import fcn_head

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def head_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def normalize(axis = -1, momentum = 0.9, epsilon = 1e-5, **kwargs):
    return tf.keras.layers.BatchNormalization(axis = axis, momentum = momentum, epsilon = epsilon, **kwargs)

def fcn(feature, n_class = 35, n_feature = 512, n_depth = 2, dropout_rate = 0.1,
        neck = None,
        logits_convolution = head_conv, logits_activation = tf.keras.activations.softmax,
        convolution = conv, normalize = normalize, activation = tf.keras.activations.relu):
    if neck is not None:
        feature = neck(name = "neck")(feature)
    out = fcn_head(feature, n_class = n_class, n_feature = n_feature, n_depth = n_depth, dropout_rate = dropout_rate,
                   logits_convolution = logits_convolution, logits_activation = logits_activation,
                   convolution = convolution, normalize = normalize, activation = activation)
    return out

def aux_fcn(feature, n_class = 35, n_feature = 256, n_depth = 1, dropout_rate = 0.1,
            logits_convolution = head_conv, logits_activation = tf.keras.activations.softmax,
            convolution = conv, normalize = normalize, activation = tf.keras.activations.relu):
    if isinstance(feature, (tuple, list)):
        feature = feature[-2]
    out = fcn_head(feature, n_class = n_class, n_feature = n_feature, n_depth = n_depth, dropout_rate = dropout_rate,
                   logits_convolution = logits_convolution, logits_activation = logits_activation,
                   convolution = convolution, normalize = normalize, activation = activation,
                   prefix = "aux")
    return out