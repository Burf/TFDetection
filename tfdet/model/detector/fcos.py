import tensorflow as tf
import numpy as np

from tfdet.core.anchor import generate_points
from ..head.fcos import CenternessNet, Scale, ClassNet, BoxNet
from ..neck import fpn

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def fcos(feature, n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4, sub_sampling = 1, centerness = True,
         sub_n_feature = None, sub_normalize = tf.keras.layers.BatchNormalization, 
         neck = fpn, neck_n_depth = 1,
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
    sub_n_feature = sub_n_feature if sub_n_feature is not None else n_feature
    
    for index in range(sub_sampling):
        x = feature[-1]
        if index == 0:
            x = tf.keras.layers.Conv2D(sub_n_feature, 1, use_bias = sub_normalize is None, name = "feature_sub_sampling_pre_conv")(x)
            if sub_normalize is not None:
                x = sub_normalize(name = "feature_sub_sampling_pre_norm")(x)
        feature.append(tf.keras.layers.MaxPooling2D((3, 3), strides = 2, padding = "same", name = "feature_sub_sampling{0}".format(index + 1) if 1 < sub_sampling else "feature_sub_sampling")(x))
    if neck_n_depth < 1:
        feature = [tf.keras.layers.Conv2D(n_feature, 1, use_bias = True, kernel_initializer = "he_normal", name = "feature_resample_conv{0}".format(i + 1) if 1 < len(feature) else "feature_resample_conv")(x) for i, x in enumerate(feature)]
    else:
        for index in range(neck_n_depth):
            feature = neck(name = "feature_pyramid_network{0}".format(index + 1) if 1 < neck_n_depth else "feature_pyramid_network")(feature)
    
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