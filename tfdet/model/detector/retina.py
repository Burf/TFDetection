import tensorflow as tf
import numpy as np

from tfdet.core.anchor import generate_anchors
from ..head.retina import ClassNet, BoxNet
from ..neck import FeatureAlign, fpn

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def neck(n_feature = 256, n_sampling = 1, pre_sampling = True, neck = fpn, neck_n_depth = 1, convolution = conv, normalize = tf.keras.layers.BatchNormalization, **kwargs):
    return FeatureAlign(n_feature = n_feature, n_sampling = n_sampling, pre_sampling = pre_sampling, neck = neck, neck_n_depth = neck_n_depth, convolution = convolution, normalize = normalize, **kwargs)

def retinanet(feature, n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4,
              scale = [0.03125, 0.0625, 0.125, 0.25, 0.5], ratio = [0.5, 1, 2], auto_scale = True,
              neck = neck,
              cls_convolution = conv, cls_normalize = tf.keras.layers.BatchNormalization, cls_activation = tf.keras.activations.relu, 
              box_convolution = conv, box_normalize = tf.keras.layers.BatchNormalization, box_activation = tf.keras.activations.relu):
    if tf.is_tensor(image_shape) and 2 < tf.keras.backend.ndim(image_shape) or (not tf.is_tensor(image_shape) and 2 < np.ndim(image_shape)):
        image_shape = tf.shape(image_shape) if tf.keras.backend.int_shape(image_shape)[-3] is None else tf.keras.backend.int_shape(image_shape)
    if 2 < np.shape(image_shape)[0]:
        image_shape = image_shape[-3:-1]
    if not isinstance(feature, list):
        feature = [feature]
    feature = list(feature)
    
    feature = neck(name = "neck")(feature)
        
    n_anchor = len(scale) * len(ratio)
    if isinstance(scale, list) and isinstance(scale[0], list):
        n_anchor = len(scale[0]) * len(ratio)
    elif auto_scale and (len(scale) % len(feature)) == 0:
        n_anchor = (len(scale) // len(feature)) * len(ratio)
    logits = ClassNet(n_anchor, n_class, n_feature, n_depth, convolution = cls_convolution, normalize = cls_normalize, activation = cls_activation, name = "class_net")(feature)
    regress = BoxNet(n_anchor, n_feature, n_depth, convolution = box_convolution, normalize = box_normalize, activation = box_activation, name = "box_net")(feature)
    anchors = generate_anchors(feature, image_shape, scale, ratio, normalize = True, auto_scale = auto_scale)

    valid_flags = tf.logical_and(tf.less_equal(anchors[..., 2], 1),
                                 tf.logical_and(tf.less_equal(anchors[..., 3], 1),
                                                tf.logical_and(tf.greater_equal(anchors[..., 0], 0),
                                                               tf.greater_equal(anchors[..., 1], 0))))
    #valid_indices = tf.range(tf.shape(anchors)[0])[valid_flags]
    valid_indices = tf.where(valid_flags)[:, 0]
    logits = tf.gather(logits, valid_indices, axis = 1)
    regress = tf.gather(regress, valid_indices, axis = 1)
    anchors = tf.gather(anchors, valid_indices)
    return logits, regress, anchors