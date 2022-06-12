import tensorflow as tf
import numpy as np

from ..head import retina_head
from ..neck import FeatureAlign, fpn

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def neck(n_feature = 256, n_sampling = 1, pre_sampling = True, neck = fpn, neck_n_depth = 1, convolution = conv, normalize = tf.keras.layers.BatchNormalization, **kwargs):
    return FeatureAlign(n_feature = n_feature, n_sampling = n_sampling, pre_sampling = pre_sampling, neck = neck, neck_n_depth = neck_n_depth, convolution = convolution, normalize = normalize, **kwargs)

def retinanet(feature, n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4,
              scale = [0.03125, 0.0625, 0.125, 0.25, 0.5], ratio = [0.5, 1, 2], auto_scale = True,
              neck = neck, convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    if neck is not None:
        feature = neck(name = "neck")(feature)
    logits, regress, anchors = retina_head(feature, n_class = n_class, image_shape = image_shape, n_feature = n_feature, n_depth = n_depth, scale = scale, ratio = ratio, auto_scale = auto_scale,
                                           convolution = convolution, normalize = normalize, activation = activation)
    return logits, regress, anchors