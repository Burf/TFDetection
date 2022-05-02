import tensorflow as tf
import numpy as np

from tfdet.core.anchor import generate_points
from ..head.fcos import CenternessNet, Scale, ClassNet, BoxNet
from ..neck import FeatureAlign, fpn

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def neck(n_feature = 256, n_sampling = 1, pre_sampling = True, neck = fpn, neck_n_depth = 1, convolution = conv, normalize = tf.keras.layers.BatchNormalization, **kwargs):
    return FeatureAlign(n_feature = n_feature, n_sampling = n_sampling, pre_sampling = pre_sampling, neck = neck, neck_n_depth = neck_n_depth, convolution = convolution, normalize = normalize, **kwargs)

def fcos(feature, n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4, centerness = True,
         neck = neck,
         cls_convolution = conv, cls_normalize = tf.keras.layers.BatchNormalization, cls_activation = tf.keras.activations.relu, 
         box_convolution = conv, box_normalize = tf.keras.layers.BatchNormalization, box_activation = tf.keras.activations.relu,
         centerness_logits_activation = tf.keras.activations.sigmoid, centerness_convolution = conv, centerness_normalize = None):
    if tf.is_tensor(image_shape) and 2 < tf.keras.backend.ndim(image_shape) or (not tf.is_tensor(image_shape) and 2 < np.ndim(image_shape)):
        image_shape = tf.shape(image_shape) if tf.keras.backend.int_shape(image_shape)[-3] is None else tf.keras.backend.int_shape(image_shape)
    if 2 < np.shape(image_shape)[0]:
        image_shape = image_shape[-3:-1]
    if not isinstance(feature, list):
        feature = [feature]
    feature = list(feature)
    
    feature = neck(name = "neck")(feature)
    
    n_anchor = 1
    logits, logits_feature = ClassNet(n_anchor, n_class, n_feature, n_depth, convolution = cls_convolution, normalize = cls_normalize, activation = cls_activation, concat = False, name = "class_net")(feature, feature = True)
    regress = BoxNet(n_anchor, n_feature, n_depth, convolution = box_convolution, normalize = box_normalize, activation = box_activation, concat = False, name = "box_net")(feature)
    regress = Scale(1., name = "box_net_with_scale_factor")(regress)
    if not isinstance(regress, list):
        regress = [regress]
    act = tf.keras.layers.Activation(tf.exp, name = "box_net_exp_with_scale_factor")
    regress = [act(r) for r in regress]
    if len(regress) == 1:
        regress = regress[0]
    if centerness:
        centerness = CenternessNet(n_anchor, concat = False, logits_activation = centerness_logits_activation, convolution = centerness_convolution, normalize = centerness_normalize, name = "centerness_net")(logits_feature)
    else:
        centerness = None
    points = generate_points(feature, image_shape, stride = None, normalize = True, concat = False) #stride = None > Auto Stride (ex: level 3~5 + pooling 6~7 > [8, 16, 32, 64, 128], level 2~5 + pooling 6 > [4, 8, 16, 32, 64])
    result = [r for r in [logits, regress, points, centerness] if r is not None]
    return result