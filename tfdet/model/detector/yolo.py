import os
import tensorflow as tf
import numpy as np

from tfdet.core.anchor import generate_yolo_anchors
from ..backbone.darknet import darknet_conv_block, darknet53, darknet19
from ..head.yolo import yolo_conv_block, yolo_classifier

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    
def leaky_relu(x, alpha = 0.1):
    return tf.nn.leaky_relu(x, alpha = alpha)
    
def normalize(axis = -1, **kwargs):
    return tf.keras.layers.BatchNormalization(axis = axis, **kwargs)

def yolo(x, n_class = 80,
         size = [[0.01645, 0.02138], [0.02632, 0.04934], [0.05428, 0.03783],
                 [0.04934, 0.10033], [0.10197, 0.07401], [0.09704, 0.19572],
                 [0.19079, 0.14803], [0.25658, 0.32566], [0.61349, 0.53618]], 
         auto_size = True,
         tiny = False, csp = True, shared = True, method = "nearest",
         normalize = normalize, activation = mish, post_activation = leaky_relu,
         weights = "darknet"):
    image_shape = tf.shape(x)[-3:-1] if tf.keras.backend.int_shape(x)[-3] is None else tf.keras.backend.int_shape(x)[-3:-1]
    result = []
    if tiny:
        feature = darknet19(x, csp = csp, normalize = normalize, activation = activation, weights = weights)
        n_anchor = len(size)
        if isinstance(size, list) and isinstance(size[0], list) and isinstance(size[0][0], list):
            n_anchor = len(size[0][0])
        elif auto_size and (len(size) % len(feature)) == 0:
            n_anchor = len(size) // len(feature)
        anchors = generate_yolo_anchors(feature, image_shape, size, normalize = True, auto_size = auto_size)
        
        out = feature[-1]
        out = darknet_conv_block(out, 256, 1, normalize = normalize, activation = post_activation)
        score, logits, regress = yolo_classifier(out, n_class, 512, n_anchor = n_anchor, shared = shared, normalize = normalize, activation = post_activation)
        result.append([score, logits, regress])
        
        out = darknet_conv_block(out, 128, 1, normalize = normalize, activation = post_activation)
        target_size = tf.shape(feature[-2])[-3:-1]
        out = tf.image.resize(out, target_size, method = method)
        out = tf.keras.layers.Concatenate(axis = -1)([out, feature[-2]])
        
        score, logits, regress = yolo_classifier(out, n_class, 256, n_anchor = n_anchor, shared = shared, normalize = normalize, activation = post_activation)
        result.append([score, logits, regress])
        result = result[::-1]
    else:
        feature = darknet53(x, csp = csp, normalize = normalize, activation = activation, weights = weights)
        n_anchor = len(size)
        if isinstance(size, list) and isinstance(size[0], list) and isinstance(size[0][0], list):
            n_anchor = len(size[0][0])
        elif auto_size and (len(size) % len(feature)) == 0:
            n_anchor = len(size) // len(feature)
        anchors = generate_yolo_anchors(feature, image_shape, size, normalize = True, auto_size = auto_size)
        
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
                score, logits, regress = yolo_classifier(out, n_class, _n_feature * 2, n_anchor = n_anchor, shared = shared, normalize = normalize, activation = post_activation)
                result.append([score, logits, regress])
                if index < len(n_feature) - 1:
                    out = darknet_conv_block(out, _n_feature * 2, 3, stride_size = 2, normalize = normalize, activation = post_activation)
                    out = tf.keras.layers.Concatenate(axis = -1)([out, feature[index + 1]])
        else:
            n_feature = [512, 256, 128]
            for index, _n_feature in enumerate(n_feature):
                out = yolo_conv_block(out, _n_feature, normalize = normalize, activation = post_activation)
                score, logits, regress = yolo_classifier(out, n_class, _n_feature * 2, n_anchor = n_anchor, shared = shared, normalize = normalize, activation = post_activation)
                result.append([score, logits, regress])
                if index < len(n_feature) - 1:
                    out = darknet_conv_block(out, _n_feature // 2, 1, normalize = normalize, activation = post_activation)
                    target_size = tf.shape(feature[-(index + 2)])[-3:-1]
                    out = tf.image.resize(out, target_size, method = method)
                    out = tf.keras.layers.Concatenate(axis = -1)([out, feature[-(index + 2)]])
            result = result[::-1]
    result = list(zip(*result))
    score, logits, regress = [tf.keras.layers.Concatenate(axis = 1)(r) for r in result]
    
    valid_flags = tf.logical_and(tf.less_equal(anchors[..., 2], 1),
                                 tf.logical_and(tf.less_equal(anchors[..., 3], 1),
                                                tf.logical_and(tf.greater_equal(anchors[..., 0], 0),
                                                               tf.greater_equal(anchors[..., 1], 0))))
    #valid_indices = tf.range(tf.shape(anchors)[0])[valid_flags]
    valid_indices = tf.where(valid_flags)[:, 0]
    score = tf.gather(score, valid_indices, axis = 1)
    logits = tf.gather(logits, valid_indices, axis = 1)
    regress = tf.gather(regress, valid_indices, axis = 1)
    anchors = tf.gather(anchors, valid_indices)
    return score, logits, regress, anchors

yolo_urls = {"yolo_v3":"https://pjreddie.com/media/files/yolov3.weights",
             "yolo_tiny_v3":"https://pjreddie.com/media/files/yolov3-tiny.weights",
             "yolo_v4":"https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights",
             "yolo_tiny_v4":"https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"}

def load_weight(model, path, n_class = 80, only_darknet = True):
    """
    https://pjreddie.com/media/files/yolov3.weights
    https://pjreddie.com/media/files/yolov3-tiny.weights
    https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights
    https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights
    """
    path = tf.keras.utils.get_file(fname = os.path.basename(path), origin = path, cache_subdir = "models")

    etc_names = []
    head_layers = []
    if only_darknet:
        layers = []
        for l in model.layers:
            if "darknet" not in l.name:
                break
            elif 0 < len(l.weights):
                layers.append(l)
    else:
        layers = [l for l in model.layers if 0 < len(l.weights)]
        head_cnt = -(len(layers) - len([l for l in layers if "conv" in l.name]) * 2)
        if 0 < head_cnt:
            head_layers = layers[-head_cnt:]
            head_feature = head_layers[0].filters
            if 1 < (head_cnt / 3):
                etc_names = [l.name for l in head_layers[int(head_cnt / 3):]]
                head_feature = sum([head_layers[i * int(head_cnt / 3)].filters for i in range(3)])
            n_anchors = int(head_feature / (n_class + 5))
    head_names = [l.name for l in head_layers]
    layers = [l[0] for l in sorted([[l, "{0}_{1:04d}".format(l.name.split("_")[-2], int(l.name.split("_")[-1])) if (l.name.split("_")[-1][0] not in ["c", "n"]) else l.name.split("_")[-1]] for l in layers if l.name not in etc_names], key = lambda x: x[1])]
    head_layers = [l[0] for l in sorted([[l, "{0}_{1:04d}".format(l.name.split("_")[-2], int(l.name.split("_")[-1]))] for l in head_layers], key = lambda x: x[1])]

    convs = []
    bns = []
    head_convs = []
    with open(path, "rb") as file:
        major, minor, revision, seen, _ = np.fromfile(file, dtype = np.int32, count = 5)
        for l in [l for l in layers if "conv" in l.name]:
            n_filter = l.filters
            kernel_size = l.kernel_size[0]
            input_filter = l.input_shape[-1]
            
            if l.name not in head_names:
                bn = np.fromfile(file, dtype = np.float32, count = 4 * n_filter)
                bn = bn.reshape((4, n_filter))[[1, 0, 2, 3]] #gamma, beta, mean, variance
            else:
                n_filter = 255 #3 * (80 + 5)
                bias = np.fromfile(file, dtype = np.float32, count = n_filter)
            
            conv = np.fromfile(file, dtype = np.float32, count = np.product([n_filter, input_filter, kernel_size, kernel_size]))
            conv = conv.reshape([n_filter, input_filter, kernel_size, kernel_size]).transpose([2, 3, 1, 0]) #h, w, input_dim, output_dim
            
            if l.name not in head_names:
                bns.append(bn)
                convs.append(conv)
            else:
                conv_rsl = tf.split(conv, num_or_size_splits = [12, 3, 240], axis = -1)
                bias_rsl = tf.split(bias, num_or_size_splits = [12, 3, 240], axis = -1)
                
                conv_regress = tf.reshape(tf.reshape(conv_rsl[0], [1, 1, -1, 3, 4])[:, :, :, :n_anchors], [1, 1, -1, n_anchors * 4])
                conv_score = tf.reshape(tf.reshape(conv_rsl[1], [1, 1, -1, 3])[:, :, :, :n_anchors], [1, 1, -1, n_anchors])
                conv_logits = tf.reshape(tf.reshape(conv_rsl[2], [1, 1, -1, 3, 80])[:, :, :, :n_anchors, :n_class], [1, 1, -1, n_anchors * n_class])
                bias_regress = tf.reshape(tf.reshape(bias_rsl[0], [3, 4])[:n_anchors], [-1])
                bias_score = bias_rsl[1][:n_anchors]
                bias_logits = tf.reshape(tf.reshape(bias_rsl[2], [3, 80])[:n_anchors, :n_class], [-1])
                
                if 1 < (head_cnt / 3):
                    head_convs.append([conv_score, bias_score])
                    head_convs.append([conv_logits, bias_logits])
                    head_convs.append([conv_regress, bias_regress])
                else:
                    conv = tf.concat([conv_regress, conv_score, conv_logits], axis = -1)
                    bias = tf.concat([bias_regress, bias_score, bias_logits], axis = -1)
                    head_convs.append([conv, bias])
    for l in layers:
        if l.name not in head_names:
            if "conv" in l.name:
                l.set_weights([convs.pop(0)])
            else:
                l.set_weights(bns.pop(0))
    for l in head_layers:
        l.set_weights(head_convs.pop(0))
    return model

def yolo_v3(x, n_class = 80, size = [[0.01645, 0.02138], [0.02632, 0.04934], [0.05428, 0.03783],
                                     [0.04934, 0.10033], [0.10197, 0.07401], [0.09704, 0.19572],
                                     [0.19079, 0.14803], [0.25658, 0.32566], [0.61349, 0.53618]], 
            auto_size = True, shared = True, method = "nearest",
            normalize = normalize, activation = mish, post_activation = leaky_relu, weights = "darknet"):
    out = yolo(x, n_class, size = size, auto_size = auto_size, tiny = False, csp = False, shared = shared, method = method, normalize = normalize, activation = activation, post_activation = post_activation, weights = "darknet" if weights == "darknet" else None)
    
    if weights is not None and weights != "darknet":
        model = tf.keras.Model(x, out[:-1])
        if weights == "yolo":
            load_weight(model, yolo_urls["yolo_v3"], n_class = n_class, only_darknet = False)
        else:
            model.load_weights(weights)
    return out

def yolo_tiny_v3(x, n_class = 80, size = [[0.05529, 0.06490], [0.08894, 0.13942], [0.19471, 0.19712],
                                          [0.19471, 0.19712], [0.32452, 0.40625], [0.82692, 0.76683]],
                 auto_size = True, shared = True, method = "nearest",
                 normalize = normalize, activation = mish, post_activation = leaky_relu, weights = "darknet"):
    out = yolo(x, n_class, size = size, auto_size = auto_size, tiny = True, csp = False, shared = shared, method = method, normalize = normalize, activation = activation, post_activation = post_activation, weights = "darknet" if weights == "darknet" else None)
    
    if weights is not None and weights != "darknet":
        model = tf.keras.Model(x, out[:-1])
        if weights == "yolo":
            load_weight(model, yolo_urls["yolo_tiny_v3"], n_class = n_class, only_darknet = False)
        else:
            model.load_weights(weights)
    return out

def yolo_v4(x, n_class = 80, size = [[0.01645, 0.02138], [0.02632, 0.04934], [0.05428, 0.03783],
                                     [0.04934, 0.10033], [0.10197, 0.07401], [0.09704, 0.19572],
                                     [0.19079, 0.14803], [0.25658, 0.32566], [0.61349, 0.53618]], 
            auto_size = True, shared = True, method = "nearest",
            normalize = normalize, activation = mish, post_activation = leaky_relu, weights = "darknet"):
    out = yolo(x, n_class, size = size, auto_size = auto_size, tiny = False, csp = True, shared = shared, method = method, normalize = normalize, activation = activation, post_activation = post_activation, weights = "darknet" if weights == "darknet" else None)
    
    if weights is not None and weights != "darknet":
        model = tf.keras.Model(x, out[:-1])
        if weights == "yolo":
            load_weight(model, yolo_urls["yolo_v4"], n_class = n_class, only_darknet = False)
        else:
            model.load_weights(weights)
    return out

def yolo_tiny_v4(x, n_class = 80, size = [[0.05529, 0.06490], [0.08894, 0.13942], [0.19471, 0.19712],
                                          [0.19471, 0.19712], [0.32452, 0.40625], [0.82692, 0.76683]],
                 auto_size = True, shared = True, method = "nearest",
                 normalize = normalize, activation = mish, post_activation = leaky_relu, weights = "darknet"):
    out = yolo(x, n_class, size = size, auto_size = auto_size, tiny = True, csp = True, shared = shared, method = method, normalize = normalize, activation = activation, post_activation = post_activation, weights = "darknet" if weights == "darknet" else None)
    
    if weights is not None and weights != "darknet":
        model = tf.keras.Model(x, out[:-1])
        if weights == "yolo":
            load_weight(model, yolo_urls["yolo_tiny_v4"], n_class = n_class, only_darknet = False)
        else:
            model.load_weights(weights)
    return out