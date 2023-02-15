import tensorflow as tf
import numpy as np

from tfdet.core.ops.initializer import PriorProbability
from ..head import fcos_head
from ..neck import FeatureAlign, fpn

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def neck_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "glorot_uniform", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def cls_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), bias_initializer = PriorProbability(probability = 0.01), **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def bbox_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def conf_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def group_normalize(groups = 32, epsilon = 1e-5, momentum = 0.9, **kwargs):
    layer = None
    try:
        layer = tf.keras.layers.GroupNormalization(groups = groups, epsilon = epsilon, **kwargs)
    except:
        try:
            import tensorflow_addons as tfa
            layer = tfa.layers.GroupNormalization(groups = groups, epsilon = epsilon, **kwargs)
        except:
            pass
    if layer is None:
        print("If you want to use 'GroupNormalization', please install 'tensorflow 2.11▲ or tensorflow_addons'")
        layer = tf.keras.layers.BatchNormalization(momentum = momentum, epsilon = epsilon, **kwargs)
    return layer

def neck(n_feature = 256, n_sampling = 1, pre_sampling = True, neck = fpn, neck_n_depth = 1, use_bias = None, convolution = neck_conv, normalize = None, **kwargs):
    return FeatureAlign(n_feature = n_feature, n_sampling = n_sampling, pre_sampling = pre_sampling, neck = neck, neck_n_depth = neck_n_depth, use_bias = use_bias, convolution = convolution, normalize = normalize, **kwargs)

def fcos(feature, n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4, use_bias = None,
         centerness = True,
         neck = neck,
         cls_convolution = cls_conv, cls_activation = tf.keras.activations.sigmoid,
         bbox_convolution = bbox_conv, bbox_activation = tf.keras.activations.linear,
         conf_convolution = conf_conv, conf_normalize = None, conf_activation = tf.keras.activations.sigmoid,
         convolution = conv, normalize = group_normalize, activation = tf.keras.activations.relu):
    if neck is not None:
        feature = neck(name = "neck")(feature)
    out = fcos_head(feature, n_class = n_class, image_shape = image_shape, n_feature = n_feature, n_depth = n_depth, use_bias = use_bias,
                    centerness = centerness,
                    cls_convolution = cls_convolution, cls_activation = cls_activation,
                    bbox_convolution = bbox_convolution, bbox_activation = bbox_activation,
                    conf_convolution = conf_convolution, conf_normalize = conf_normalize, conf_activation = conf_activation,
                    convolution = convolution, normalize = normalize, activation = activation)
    return out