import tensorflow as tf
import numpy as np

from ..head import fcos_head
from ..neck import FeatureAlign, fpn

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def neck(n_feature = 256, n_sampling = 1, pre_sampling = True, neck = fpn, neck_n_depth = 1, convolution = conv, normalize = tf.keras.layers.BatchNormalization, **kwargs):
    return FeatureAlign(n_feature = n_feature, n_sampling = n_sampling, pre_sampling = pre_sampling, neck = neck, neck_n_depth = neck_n_depth, convolution = convolution, normalize = normalize, **kwargs)

def fcos(feature, n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4, centerness = True,
         neck = neck,
         convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu, 
         centerness_convolution = conv, centerness_normalize = None, centerness_activation = tf.keras.activations.sigmoid):
    if neck is not None:
        feature = neck(name = "neck")(feature)
    outs = fcos_head(feature, n_class = n_class, image_shape = image_shape, n_feature = n_feature, n_depth = n_depth, centerness = centerness,
                     convolution = convolution, normalize = normalize, activation = activation,
                     centerness_convolution = centerness_convolution, centerness_normalize = centerness_normalize, centerness_activation = centerness_activation)
    return outs #[r for r in [logits, regress, points, centerness] if r is not None]