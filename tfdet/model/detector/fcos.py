import tensorflow as tf
import numpy as np

from tfdet.core.util.anchor import generate_points
from ..head.fcos import CenternessNet, Scale, ClassNet, BoxNet
from ..neck import fpn

def fcos(feature, n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4, sub_sampling = 1, centerness = True,
         sub_n_feature = None, sub_momentum = 0.997, sub_epsilon = 1e-4, fpn = fpn, fpn_n_depth = 1,
         cls_activation = tf.keras.activations.relu, bbox_activation = tf.keras.activations.relu,
         **kwargs):
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
            x = tf.keras.layers.Conv2D(sub_n_feature, 1, use_bias = False, name = "feature_sub_sampling_pre_conv")(x)
            x = tf.keras.layers.BatchNormalization(axis = -1, momentum = sub_momentum, epsilon = sub_epsilon, name = "feature_sub_sampling_pre_norm")(x)
        feature.append(tf.keras.layers.MaxPooling2D((3, 3), strides = 2, padding = "same", name = "feature_sub_sampling{0}".format(index + 1) if 1 < sub_sampling else "feature_sub_sampling")(x))
    if fpn_n_depth < 1:
        feature = [tf.keras.layers.Conv2D(n_feature, 1, use_bias = True, kernel_initializer = "he_normal", name = "feature_resample_conv{0}".format(i + 1) if 1 < len(feature) else "feature_resample_conv")(x) for i, x in enumerate(feature)]
    else:
        for index in range(fpn_n_depth):
            feature = fpn(name = "feature_pyramid_network{0}".format(index + 1) if 1 < fpn_n_depth else "feature_pyramid_network")(feature)
    
    n_anchor = 1
    logits, logits_feature = ClassNet(n_anchor, n_class, n_feature, n_depth, cls_activation, concat = False, name = "class_net")(feature, feature = True)
    regress = BoxNet(n_anchor, n_feature, n_depth, bbox_activation, concat = False, name = "box_net")(feature)
    regress = Scale(1., name = "box_net_with_scale_factor")(regress)
    if not isinstance(regress, list):
        regress = [regress]
    act = tf.keras.layers.Activation(tf.exp, name = "box_net_exp_with_scale_factor")
    regress = [act(r) for r in regress]
    if len(regress) == 1:
        regress = regress[0]
    if centerness:
        centerness = CenternessNet(n_anchor, concat = False, name = "centerness_net")(logits_feature)
    else:
        centerness = None
    points = generate_points(feature, image_shape, stride = None, normalize = True, concat = False) #stride = None > Auto Stride (ex: level 3~5 + pooling 6~7 > [8, 16, 32, 64, 128], level 2~5 + pooling 6 > [4, 8, 16, 32, 64])
    result = [r for r in [logits, regress, points, centerness] if r is not None]
    return result