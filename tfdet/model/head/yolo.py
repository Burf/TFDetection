import tensorflow as tf
import numpy as np

from tfdet.core.anchor import generate_yolo_anchors
from ..backbone.darknet import darknet_conv_block, darknet53, darknet19

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def leaky_relu(x, alpha = 0.1):
    return tf.nn.leaky_relu(x, alpha = alpha)
    
def yolo_conv_block(x, n_feature, normalize = tf.keras.layers.BatchNormalization, activation = leaky_relu):
    out = darknet_conv_block(x, n_feature, 1, normalize = normalize, activation = activation)
    out = darknet_conv_block(out, n_feature * 2, 3, normalize = normalize, activation = activation)
    out = darknet_conv_block(out, n_feature, 1, normalize = normalize, activation = activation)
    out = darknet_conv_block(out, n_feature * 2, 3, normalize = normalize, activation = activation)
    out = darknet_conv_block(out, n_feature, 1, normalize = normalize, activation = activation)
    return out

def yolo_classifier(x, n_class, n_feature, n_anchor = 3, feature_share = True, normalize = tf.keras.layers.BatchNormalization, activation = leaky_relu):
    out = darknet_conv_block(x, n_feature, 3, normalize = normalize, activation = activation)
    if feature_share:
        out = darknet_conv_block(out, n_anchor * (n_class + 5), 1, normalize = None, activation = None)
        out = tf.keras.layers.Reshape((-1, (n_class + 5)))(out)
        regress, score, logits = tf.split(out, num_or_size_splits = [4, 1, n_class], axis = -1)
    else:
        score = darknet_conv_block(out, n_anchor * 1, 1, normalize = None, activation = None)
        logits = darknet_conv_block(out, n_anchor * n_class, 1, normalize = None, activation = None)
        regress = darknet_conv_block(out, n_anchor * 4, 1, normalize = None, activation = None)
        score = tf.keras.layers.Reshape((-1, 1))(score)
        logits = tf.keras.layers.Reshape((-1, n_class))(logits)
        regress = tf.keras.layers.Reshape((-1, 4))(regress)
    xy, wh = tf.split(regress, num_or_size_splits = [2, 2], axis = -1)
    xy = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(xy)
    score = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(score)
    logits = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(logits)
    regress = tf.keras.layers.Concatenate(axis = -1)([xy, wh])
    return score, logits, regress

def yolo_head(feature, n_class = 80, image_shape = [608, 608],
              size = [[0.01645, 0.02138], [0.02632, 0.04934], [0.05428, 0.03783],
                      [0.04934, 0.10033], [0.10197, 0.07401], [0.09704, 0.19572],
                      [0.19079, 0.14803], [0.25658, 0.32566], [0.61349, 0.53618]], 
              auto_size = True,
              tiny = False, csp = True, feature_share = True, method = "nearest",
              normalize = tf.keras.layers.BatchNormalization, activation = mish, post_activation = leaky_relu):
    if tf.is_tensor(image_shape) and 2 < tf.keras.backend.ndim(image_shape) or (not tf.is_tensor(image_shape) and 2 < np.ndim(image_shape)):
        image_shape = tf.shape(image_shape) if tf.keras.backend.int_shape(image_shape)[-3] is None else tf.keras.backend.int_shape(image_shape)
    if 2 < np.shape(image_shape)[0]:
        image_shape = image_shape[-3:-1]
    if not isinstance(feature, list):
        feature = [feature]
    feature = list(feature)
    
    result = []
    if tiny:
        #feature = darknet19(x, csp = csp, normalize = normalize, activation = activation, weights = None)
        out = feature[-1]
        out = darknet_conv_block(out, 256, 1, normalize = normalize, activation = post_activation)
        score, logits, regress = yolo_classifier(out, n_class, 512, n_anchor = n_anchor, feature_share = feature_share, normalize = normalize, activation = post_activation)
        result.append([score, logits, regress])
        
        out = darknet_conv_block(out, 128, 1, normalize = normalize, activation = post_activation)
        target_size = tf.shape(feature[-2])[-3:-1]
        out = tf.image.resize(out, target_size, method = method)
        out = tf.keras.layers.Concatenate(axis = -1)([out, feature[-2]])
        
        score, logits, regress = yolo_classifier(out, n_class, 256, n_anchor = n_anchor, feature_share = feature_share, normalize = normalize, activation = post_activation)
        result.append([score, logits, regress])
        result = result[::-1]
    else:
        #feature = darknet53(x, csp = csp, normalize = normalize, activation = activation, weights = None)
        out = feature[-1]
        if csp:
            n_feature = [256, 128]
            for index, _n_feature in enumerate(n_feature):
                out = darknet_conv_block(out, _n_feature, 1, normalize = normalize, activation = post_activation)
                target_size = tf.shape(feature[-(index + 2)])[-3:-1]
                out = tf.image.resize(out, target_size, method = method)
                feature[-(index + 2)] = darknet_conv_block(feature[-(index + 2)], _n_feature, 1, normalize = normalize, activation = post_activation)
                out = tf.keras.layers.Concatenate(axis = -1)([out, feature[-(index + 2)]])
                if index < len(n_feature) - 1:
                    feature[-(index + 2)] = out = yolo_conv_block(out, _n_feature, normalize = normalize, activation = post_activation)
            
            n_feature = [128, 256, 512]
            for index, _n_feature in enumerate(n_feature):
                out = yolo_conv_block(out, _n_feature, normalize = normalize, activation = post_activation)
                score, logits, regress = yolo_classifier(out, n_class, _n_feature * 2, n_anchor = n_anchor, feature_share = feature_share, normalize = normalize, activation = post_activation)
                result.append([score, logits, regress])
                if index < len(n_feature) - 1:
                    out = darknet_conv_block(out, _n_feature * 2, 3, stride_size = 2, normalize = normalize, activation = post_activation)
                    out = tf.keras.layers.Concatenate(axis = -1)([out, feature[index + 1]])
        else:
            n_feature = [512, 256, 128]
            for index, _n_feature in enumerate(n_feature):
                out = yolo_conv_block(out, _n_feature, normalize = normalize, activation = post_activation)
                score, logits, regress = yolo_classifier(out, n_class, _n_feature * 2, n_anchor = n_anchor, feature_share = feature_share, normalize = normalize, activation = post_activation)
                result.append([score, logits, regress])
                if index < len(n_feature) - 1:
                    out = darknet_conv_block(out, _n_feature // 2, 1, normalize = normalize, activation = post_activation)
                    target_size = tf.shape(feature[-(index + 2)])[-3:-1]
                    out = tf.image.resize(out, target_size, method = method)
                    out = tf.keras.layers.Concatenate(axis = -1)([out, feature[-(index + 2)]])
            result = result[::-1]
    result = list(zip(*result))
    score, logits, regress = [tf.keras.layers.Concatenate(axis = 1)(r) for r in result]
    
    n_anchor = len(size)
    if isinstance(size, list) and isinstance(size[0], list) and isinstance(size[0][0], list):
        n_anchor = len(size[0][0])
    elif auto_size and (len(size) % len(feature)) == 0:
        n_anchor = len(size) // len(feature)
    anchors = generate_yolo_anchors(feature, image_shape, size, normalize = True, auto_size = auto_size, dtype = logits.dtype)
    
    #valid_flags = tf.logical_and(tf.less_equal(anchors[..., 2], 1),
    #                             tf.logical_and(tf.less_equal(anchors[..., 3], 1),
    #                                            tf.logical_and(tf.greater_equal(anchors[..., 0], 0),
    #                                                           tf.greater_equal(anchors[..., 1], 0))))
    ##valid_indices = tf.range(tf.shape(anchors)[0])[valid_flags]
    #valid_indices = tf.where(valid_flags)[:, 0]
    #score = tf.gather(score, valid_indices, axis = 1)
    #logits = tf.gather(logits, valid_indices, axis = 1)
    #regress = tf.gather(regress, valid_indices, axis = 1)
    #anchors = tf.gather(anchors, valid_indices)
    return score, logits, regress, anchors

def yolo_v3_head(feature, n_class = 80, image_shape = [608, 608], size = [[0.01645, 0.02138], [0.02632, 0.04934], [0.05428, 0.03783],
                                                                          [0.04934, 0.10033], [0.10197, 0.07401], [0.09704, 0.19572],
                                                                          [0.19079, 0.14803], [0.25658, 0.32566], [0.61349, 0.53618]], 
                 auto_size = True, feature_share = True, method = "nearest",
                 normalize = tf.keras.layers.BatchNormalization, activation = mish, post_activation = leaky_relu):
    score, logits, regress, anchors = yolo_head(feature, n_class = n_class, image_shape = image_shape, size = size, auto_size = auto_size, tiny = False, csp = False, feature_share = feature_share, method = method)
    return score, logits, regress, anchors

def yolo_tiny_v3_head(feature, n_class = 80, image_shape = [416, 416], size = [[0.05529, 0.06490], [0.08894, 0.13942], [0.19471, 0.19712],
                                                                               [0.19471, 0.19712], [0.32452, 0.40625], [0.82692, 0.76683]],
                      auto_size = True, feature_share = True, method = "nearest",
                      normalize = tf.keras.layers.BatchNormalization, activation = mish, post_activation = leaky_relu):
    score, logits, regress, anchors = yolo_head(feature, n_class = n_class, image_shape = image_shape, size = size, auto_size = auto_size, tiny = True, csp = False, feature_share = feature_share, method = method)
    return score, logits, regress, anchors

def yolo_v4_head(feature, n_class = 80, image_shape = [608, 608], size = [[0.01645, 0.02138], [0.02632, 0.04934], [0.05428, 0.03783],
                                                                          [0.04934, 0.10033], [0.10197, 0.07401], [0.09704, 0.19572],
                                                                          [0.19079, 0.14803], [0.25658, 0.32566], [0.61349, 0.53618]], 
                 auto_size = True, feature_share = True, method = "nearest",
                 normalize = tf.keras.layers.BatchNormalization, activation = mish, post_activation = leaky_relu):
    score, logits, regress, anchors = yolo_head(feature, n_class = n_class, image_shape = image_shape, size = size, auto_size = auto_size, tiny = False, csp = True, feature_share = feature_share, method = method)
    return score, logits, regress, anchors

def yolo_tiny_v4_head(feature, n_class = 80, image_shape = [416, 416], size = [[0.05529, 0.06490], [0.08894, 0.13942], [0.19471, 0.19712],
                                                                               [0.19471, 0.19712], [0.32452, 0.40625], [0.82692, 0.76683]],
                      auto_size = True, feature_share = True, method = "nearest",
                      normalize = tf.keras.layers.BatchNormalization, activation = mish, post_activation = leaky_relu):
    score, logits, regress, anchors = yolo_head(feature, n_class = n_class, image_shape = image_shape, size = size, auto_size = auto_size, tiny = True, csp = True, feature_share = feature_share, method = method)
    return score, logits, regress, anchors

