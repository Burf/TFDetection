import os
import tensorflow as tf
import numpy as np

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    
def leaky_relu(x, alpha = 0.1):
    return tf.nn.leaky_relu(x, alpha = alpha)

def darknet_conv_block(x, n_feature, kernel_size = 3, stride_size = 1, normalize = tf.keras.layers.BatchNormalization, activation = mish):
    padding = "same"
    if stride_size != 1:
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = "valid"

    out = tf.keras.layers.Conv2D(n_feature, kernel_size, strides = stride_size, padding = padding, use_bias = normalize is None, kernel_initializer = tf.random_normal_initializer(stddev = 0.01), bias_initializer = "zeros")(x)
    if normalize is not None:
        out = normalize()(out)
    if activation is not None:
        out = tf.keras.layers.Activation(activation)(out)
    return out

def darknet_res_block(x, n_feature, reduce = True, normalize = tf.keras.layers.BatchNormalization, activation = mish):
    out = darknet_conv_block(x, n_feature // 2 if reduce else n_feature, 1, normalize = normalize, activation = activation)
    out = darknet_conv_block(out, n_feature, 3, normalize = normalize, activation = activation)
    out = tf.keras.layers.Add()([x, out])
    return out

def darknet_block(x, n_feature, n_block, stride_size = 2, csp_filter = None, reduce = True, normalize = tf.keras.layers.BatchNormalization, activation = mish):
    out = darknet_conv_block(x, n_feature, 3, stride_size = stride_size, normalize = normalize, activation = activation)
    res_filter = n_feature
    if isinstance(csp_filter, int):
        res_filter = csp_filter
        residual = darknet_conv_block(out, csp_filter, 1, normalize = normalize, activation = activation)
        out = darknet_conv_block(out, csp_filter, 1, normalize = normalize, activation = activation)
    for index in range(n_block):
        out = darknet_res_block(out, res_filter, reduce = reduce, normalize = normalize, activation = activation)
    if isinstance(csp_filter, int):
        out = darknet_conv_block(out, csp_filter, 1, normalize = normalize, activation = activation)
        out = tf.keras.layers.Concatenate(axis = -1)([out, residual])
        out = darknet_conv_block(out, n_feature, 1, normalize = normalize, activation = activation)
    return out

def darknet_tiny_block(x, n_feature, stride_size = 2, csp_filter = None, feature = False, normalize = tf.keras.layers.BatchNormalization, activation = mish):
    feat = res1 = out = darknet_conv_block(x, n_feature, 3, normalize = normalize, activation = activation)
    if isinstance(csp_filter, int):
        out = tf.split(out, num_or_size_splits = 2, axis = -1)[1]
        res2 = out = darknet_conv_block(out, csp_filter, 3, normalize = normalize, activation = activation)
        out = darknet_conv_block(out, csp_filter, 3, normalize = normalize, activation = activation)
        out = tf.keras.layers.Concatenate(axis = -1)([out, res2])
        feat = out = darknet_conv_block(out, n_feature, 1, normalize = normalize, activation = activation)
        out = tf.keras.layers.Concatenate(axis = -1)([out, res1])
    out = tf.keras.layers.MaxPool2D(2, strides = stride_size, padding = "same")(out)
    if feature:
        out = [out, feat]
    return out

def darknet53(x, csp = False, normalize = tf.keras.layers.BatchNormalization, activation = mish, post_activation = leaky_relu, weights = "darknet", indices = None):
    csp_filter = [None, None, None, None, None]
    if csp:
        csp_filter = [64, 64, 128, 256, 512]
    
    feature = []
    out = darknet_conv_block(x, 32, 3, normalize = normalize, activation = activation)
    out = darknet_block(out, 64, 1, stride_size = 2, csp_filter = csp_filter.pop(0), reduce = True, normalize = normalize, activation = activation)
    out = darknet_block(out, 128, 2, stride_size = 2, csp_filter = csp_filter.pop(0), reduce = False if csp else True, normalize = normalize, activation = activation)
    out = darknet_block(out, 256, 8, stride_size = 2, csp_filter = csp_filter.pop(0), reduce = False if csp else True, normalize = normalize, activation = activation)
    feature.append(out)
    out = darknet_block(out, 512, 8, stride_size = 2, csp_filter = csp_filter.pop(0), reduce = False if csp else True, normalize = normalize, activation = activation)
    feature.append(out)
    out = darknet_block(out, 1024, 4, stride_size = 2, csp_filter = csp_filter.pop(0), reduce = False if csp else True, normalize = normalize, activation = activation)
    if csp:
        out = darknet_conv_block(out, 512, 1, normalize = normalize, activation = post_activation)
        out = darknet_conv_block(out, 1024, 3, normalize = normalize, activation = post_activation)
        out = darknet_conv_block(out, 512, 1, normalize = normalize, activation = post_activation)
        
        #Spatial Pyramid Pooling
        pool1 = tf.keras.layers.MaxPool2D(13, strides = 1, padding = "same")(out)
        pool2 = tf.keras.layers.MaxPool2D(9, strides = 1, padding = "same")(out)
        pool3 = tf.keras.layers.MaxPool2D(5, strides = 1, padding = "same")(out)
        out = tf.keras.layers.Concatenate(axis = -1)([pool1, pool2, pool3, out])
        
        out = darknet_conv_block(out, 512, 1, normalize = normalize, activation = post_activation)
        out = darknet_conv_block(out, 1024, 3, normalize = normalize, activation = post_activation)
        out = darknet_conv_block(out, 512, 1, normalize = normalize, activation = post_activation)
    feature.append(out)
    
    if weights is not None:
        model = tf.keras.Model(x, feature)
        if weights == "darknet":
            load_weight(model, darknet_urls["{0}darknet53".format("csp_" if csp else "")], only_darknet = True)
        else:
            model.load_weights(weights)
    
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def darknet19(x, csp = False, normalize = tf.keras.layers.BatchNormalization, activation = mish, weights = "darknet", indices = None):
    if csp:
        n_feature = [64, 128, 256]
        csp_filter = [32, 64, 128]
        stride_size = [2, 2, 2]
        x = darknet_conv_block(x, 32, 3, stride_size = 2, normalize = normalize, activation = activation)
        x = darknet_conv_block(x, 64, 3, stride_size = 2, normalize = normalize, activation = activation)
    else:
        n_feature = [16, 32, 64, 128, 256, 512]
        csp_filter = [None, None, None, None, None, None]
        stride_size = [2, 2, 2, 2, 2, 1]
    
    feature = []
    out = x
    for index, (_n_feature, _csp_filter, _stride_size) in enumerate(zip(n_feature, csp_filter, stride_size)):
        out, feat = darknet_tiny_block(out, _n_feature, stride_size = _stride_size, csp_filter = _csp_filter, feature = True, normalize = normalize, activation = activation)
        feature.append(feat)
    
    if csp:
        out = darknet_conv_block(out, 512, 3, normalize = normalize, activation = activation)
        feature = [feature[-1], out]
    else:
        out = darknet_conv_block(out, 1024, 3, normalize = normalize, activation = activation)
        feature = [feature[-2], out]
    
    if weights is not None:
        model = tf.keras.Model(x, feature)
        if weights == "darknet":
            load_weight(model, darknet_urls["{0}darknet19".format("csp_" if csp else "")], only_darknet = True)
        else:
            model.load_weights(weights)
            
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature
    
def csp_darknet53(x, csp = True, normalize = tf.keras.layers.BatchNormalization, activation = mish, post_activation = leaky_relu, weights = "darknet", indices = None):
    return darknet53(x, csp = csp, normalize = normalize, activation = activation, post_activation = post_activation, weights = weights)
    
def csp_darknet19(x, csp = True, normalize = tf.keras.layers.BatchNormalization, activation = mish, weights = "darknet", indices = None):
    return darknet19(x, csp = csp, normalize = normalize, activation = activation, weights = weights)
    
darknet_urls = {"darknet53":"https://pjreddie.com/media/files/yolov3.weights",
                "darknet19":"https://pjreddie.com/media/files/yolov3-tiny.weights",
                "csp_darknet53":"https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights",
                "csp_darknet19":"https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"}

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