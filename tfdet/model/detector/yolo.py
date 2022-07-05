import os
import tensorflow as tf
import numpy as np

from ..backbone.darknet import darknet53, darknet19, csp_darknet53, csp_darknet19, load_weight
from ..head import yolo_v3_head, yolo_tiny_v3_head, yolo_v4_head, yolo_tiny_v4_head

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    
def leaky_relu(x, alpha = 0.1):
    return tf.nn.leaky_relu(x, alpha = alpha)

yolo_urls = {"yolo_v3":"https://pjreddie.com/media/files/yolov3.weights",
             "yolo_tiny_v3":"https://pjreddie.com/media/files/yolov3-tiny.weights",
             "yolo_v4":"https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights",
             "yolo_tiny_v4":"https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"}

def yolo_v3(x, n_class = 80, size = [[ 10, 13], [ 16,  30], [ 33,  23],
                                     [ 30, 61], [ 62,  45], [ 59, 119],
                                     [116, 90], [156, 198], [373, 326]],
            feature_share = True, method = "nearest",
            normalize = tf.keras.layers.BatchNormalization, activation = mish, post_activation = leaky_relu, weights = "darknet"):
    feature = darknet53(x, csp = False, normalize = normalize, activation = activation, post_activation = post_activation, weights = None)
    score, logits, regress, anchors = yolo_v3_head(feature, n_class = n_class, image_shape = tf.shape(x)[1:3], size = size, feature_share = feature_share, method = method, normalize = normalize, activation = activation, post_activation = post_activation)
    
    if weights is not None:
        model = tf.keras.Model(x, [score, logits, regress])
        if weights in ["darknet", "yolo"]:
            load_weight(model, yolo_urls["yolo_v3"], n_class = n_class, only_darknet = weights == "darknet")
        else:
            model.load_weights(weights)
    return score, logits, regress, anchors

def yolo_tiny_v3(x, n_class = 80, size = [[23, 27], [ 37,  58], [ 81,  82],
                                          [81, 82], [135, 169], [344, 319]],
                 feature_share = True, method = "nearest",
                 normalize = tf.keras.layers.BatchNormalization, activation = mish, post_activation = leaky_relu, weights = "darknet"):
    feature = darknet19(x, csp = False, normalize = normalize, activation = activation, weights = None)
    score, logits, regress, anchors = yolo_tiny_v3_head(feature, n_class = n_class, image_shape = tf.shape(x)[1:3], size = size, feature_share = feature_share, method = method, normalize = normalize, activation = activation, post_activation = post_activation)
    
    if weights is not None:
        model = tf.keras.Model(x, [score, logits, regress])
        if weights in ["darknet", "yolo"]:
            load_weight(model, yolo_urls["yolo_tiny_v3"], n_class = n_class, only_darknet = weights == "darknet")
        else:
            model.load_weights(weights)
    return score, logits, regress, anchors

def yolo_v4(x, n_class = 80, size = [[ 10, 13], [ 16,  30], [ 33,  23],
                                     [ 30, 61], [ 62,  45], [ 59, 119],
                                     [116, 90], [156, 198], [373, 326]],
            feature_share = True, method = "nearest",
            normalize = tf.keras.layers.BatchNormalization, activation = mish, post_activation = leaky_relu, weights = "darknet"):
    feature = darknet53(x, csp = True, normalize = normalize, activation = activation, post_activation = post_activation, weights = None)
    score, logits, regress, anchors = yolo_v4_head(feature, n_class = n_class, image_shape = tf.shape(x)[1:3], size = size, feature_share = feature_share, method = method, normalize = normalize, activation = activation, post_activation = post_activation)
    
    if weights is not None:
        model = tf.keras.Model(x, [score, logits, regress])
        if weights in ["darknet", "yolo"]:
            load_weight(model, yolo_urls["yolo_v4"], n_class = n_class, only_darknet = weights == "darknet")
        else:
            model.load_weights(weights)
    return score, logits, regress, anchors

def yolo_tiny_v4(x, n_class = 80, size = [[23, 27], [ 37,  58], [ 81,  82],
                                          [81, 82], [135, 169], [344, 319]],
                 feature_share = True, method = "nearest",
                 normalize = tf.keras.layers.BatchNormalization, activation = mish, post_activation = leaky_relu, weights = "darknet"):
    feature = darknet19(x, csp = True, normalize = normalize, activation = activation, weights = None)
    score, logits, regress, anchors = yolo_tiny_v4_head(feature, n_class = n_class, image_shape = tf.shape(x)[1:3], size = size, feature_share = feature_share, method = method, normalize = normalize, activation = activation, post_activation = post_activation)
    
    if weights is not None:
        model = tf.keras.Model(x, [score, logits, regress])
        if weights in ["darknet", "yolo"]:
            load_weight(model, yolo_urls["yolo_tiny_v4"], n_class = n_class, only_darknet = weights == "darknet")
        else:
            model.load_weights(weights)
    return score, logits, regress, anchors