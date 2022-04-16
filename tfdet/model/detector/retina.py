import tensorflow as tf
import numpy as np

from tfdet.core.anchor import generate_anchors
from ..head.retina import ClassNet, BoxNet
from ..neck import fpn

def retinanet(feature, n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4, sub_sampling = 1,
              scale = [0.03125, 0.0625, 0.125, 0.25, 0.5], ratio = [0.5, 1, 2], auto_scale = True,
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
        
    n_anchor = len(scale) * len(ratio)
    if isinstance(scale, list) and isinstance(scale[0], list):
        n_anchor = len(scale[0]) * len(ratio)
    elif auto_scale and (len(scale) % len(feature)) == 0:
        n_anchor = (len(scale) // len(feature)) * len(ratio)
    logits = ClassNet(n_anchor, n_class, n_feature, n_depth, cls_activation, name = "class_net")(feature)
    regress = BoxNet(n_anchor, n_feature, n_depth, bbox_activation, name = "box_net")(feature)
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