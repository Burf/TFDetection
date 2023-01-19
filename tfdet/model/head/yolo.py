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
    xy = tf.keras.layers.Activation(tf.keras.activations.sigmoid, dtype = tf.float32)(xy)
    score = tf.keras.layers.Activation(tf.keras.activations.sigmoid, dtype = tf.float32)(score)
    logits = tf.keras.layers.Activation(tf.keras.activations.sigmoid, dtype = tf.float32)(logits)
    regress = tf.keras.layers.Concatenate(axis = -1)([xy, wh])
    regress = tf.keras.layers.Activation(tf.keras.activations.linear, dtype = tf.float32)(regress)
    return score, logits, regress

def yolo_head(feature, n_class = 80, image_shape = [608, 608],
              size = [[ 10, 13], [ 16,  30], [ 33,  23],
                      [ 30, 61], [ 62,  45], [ 59, 119],
                      [116, 90], [156, 198], [373, 326]],
              tiny = False, csp = True, feature_share = True, method = "nearest",
              normalize = tf.keras.layers.BatchNormalization, activation = mish, post_activation = leaky_relu):
    if tf.is_tensor(image_shape) and 2 < tf.keras.backend.ndim(image_shape) or (not tf.is_tensor(image_shape) and 2 < np.ndim(image_shape)):
        image_shape = tf.shape(image_shape) if tf.keras.backend.int_shape(image_shape)[-3] is None else tf.keras.backend.int_shape(image_shape)
    if 2 < np.shape(image_shape)[0]:
        image_shape = image_shape[-3:-1]
    if not isinstance(feature, list):
        feature = [feature]
    if np.ndim(size) == 0:
        size = [[size, size]]
    elif np.ndim(size) == 1:
        size = np.expand_dims(size, axis = -1)
    feature = list(feature)
    
    n_anchor = len(size)
    if (len(size) % len(feature)) == 0:
        n_anchor = len(size) // len(feature)
        
    out = feature[-1]
    result = []
    if tiny:
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
    score, logits, regress = [tf.keras.layers.Concatenate(axis = 1, dtype = tf.float32)(r) for r in result]
    anchors = generate_yolo_anchors(feature, image_shape, size, normalize = True, auto_size = True, dtype = tf.float32)
    
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

def yolo_v3_head(feature, n_class = 80, image_shape = [608, 608], size = [[ 10, 13], [ 16,  30], [ 33,  23],
                                                                          [ 30, 61], [ 62,  45], [ 59, 119],
                                                                          [116, 90], [156, 198], [373, 326]],
                 feature_share = True, method = "nearest",
                 normalize = tf.keras.layers.BatchNormalization, activation = mish, post_activation = leaky_relu):
    score, logits, regress, anchors = yolo_head(feature, n_class = n_class, image_shape = image_shape, size = size, tiny = False, csp = False, feature_share = feature_share, method = method, normalize = normalize, activation = activation, post_activation = post_activation)
    return score, logits, regress, anchors

def yolo_tiny_v3_head(feature, n_class = 80, image_shape = [416, 416], size = [[23, 27], [ 37,  58], [ 81,  82],
                                                                               [81, 82], [135, 169], [344, 319]],
                      feature_share = True, method = "nearest",
                      normalize = tf.keras.layers.BatchNormalization, activation = mish, post_activation = leaky_relu):
    score, logits, regress, anchors = yolo_head(feature, n_class = n_class, image_shape = image_shape, size = size, tiny = True, csp = False, feature_share = feature_share, method = method, normalize = normalize, activation = activation, post_activation = post_activation)
    return score, logits, regress, anchors

def yolo_v4_head(feature, n_class = 80, image_shape = [608, 608], size = [[ 10, 13], [ 16,  30], [ 33,  23],
                                                                          [ 30, 61], [ 62,  45], [ 59, 119],
                                                                          [116, 90], [156, 198], [373, 326]],
                 feature_share = True, method = "nearest",
                 normalize = tf.keras.layers.BatchNormalization, activation = mish, post_activation = leaky_relu):
    score, logits, regress, anchors = yolo_head(feature, n_class = n_class, image_shape = image_shape, size = size, tiny = False, csp = True, feature_share = feature_share, method = method, normalize = normalize, activation = activation, post_activation = post_activation)
    return score, logits, regress, anchors

def yolo_tiny_v4_head(feature, n_class = 80, image_shape = [416, 416], size = [[23, 27], [ 37,  58], [ 81,  82],
                                                                               [81, 82], [135, 169], [344, 319]],
                      feature_share = True, method = "nearest",
                      normalize = tf.keras.layers.BatchNormalization, activation = mish, post_activation = leaky_relu):
    score, logits, regress, anchors = yolo_head(feature, n_class = n_class, image_shape = image_shape, size = size, tiny = True, csp = True, feature_share = feature_share, method = method, normalize = normalize, activation = activation, post_activation = post_activation)
    return score, logits, regress, anchors

