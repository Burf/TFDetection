#https://github.com/Burf/ResNeSt-Tensorflow2
import traceback

import tensorflow as tf

def group_conv(x, filters = None, kernel_size = 3, **kwargs):
    if not isinstance(kernel_size, list):
        kernel_size = [kernel_size]
    n_group = len(kernel_size)
    if n_group == 1:
        out = [x]
    else:
        size = tf.keras.backend.int_shape(x)[-1]
        split_size = [size // n_group if index != 0 else size // n_group + size % n_group for index in range(n_group)]
        out = tf.split(x, split_size, axis = -1)
    
    name = None
    if "name" in kwargs:
        name = kwargs["name"]
    result = []
    for index in range(n_group):
        kwargs["filters"] = filters // n_group
        if index == 0:
            kwargs["filters"] += filters % n_group
        kwargs["kernel_size"] = kernel_size[index]
        if name is not None and 1 < n_group:
            kwargs["name"] = "{0}_group{1}".format(name, index + 1)
        result.append(tf.keras.layers.Conv2D(**kwargs)(out[index]))
    if n_group == 1:
        out = result[0]
    else:
        out = tf.keras.layers.Concatenate(axis = -1, name = name)(result)
    return out

def split_attention_block(x, n_filter, kernel_size = 3, stride_size = 1, dilation = 1, group_size = 1, radix = 1, dropout_rate = 0., expansion = 4, prefix = ""):
    if len(prefix) != 0:
        prefix += "_"
    out = group_conv(x, n_filter * radix, [kernel_size] * (group_size * radix), strides = stride_size, dilation_rate = dilation, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "{0}split_attention_conv1".format(prefix))
    out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "{0}split_attention_bn1".format(prefix))(out)
    if 0 < dropout_rate:
        out = tf.keras.layers.Dropout(dropout_rate, name = "{0}split_attention_dropout1".format(prefix))(out)
    out = tf.keras.layers.Activation(tf.keras.activations.relu, name = "{0}split_attention_act1".format(prefix))(out)
    
    inter_channel = max(tf.keras.backend.int_shape(x)[-1] * radix // expansion, 32)
    if 1 < radix:
        split = tf.split(out, radix, axis = -1)
        out = tf.keras.layers.Add(name = "{0}split_attention_add".format(prefix))(split)
    out = tf.keras.layers.GlobalAveragePooling2D(name = "{0}split_attention_gap".format(prefix))(out)
    out = tf.keras.layers.Reshape([1, 1, n_filter], name = "{0}split_attention_expand_dims".format(prefix))(out)
    
    out = group_conv(out, inter_channel, [1] * group_size, padding = "same", use_bias = True, kernel_initializer = "he_normal", name = "{0}split_attention_conv2".format(prefix))
    out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "{0}split_attention_bn2".format(prefix))(out)
    out = tf.keras.layers.Activation("relu", name = "{0}split_attention_act2".format(prefix))(out)
    out = group_conv(out, n_filter * radix, [1] * group_size, padding = "same", use_bias = True, kernel_initializer = "he_normal", name = "{0}split_attention_conv3".format(prefix))
    
    #attention = rsoftmax(out, n_filter, radix, group_size)
    attention = rSoftMax(n_filter, radix, group_size, name = "{0}split_attention_softmax".format(prefix))(out)
    if 1 < radix:
        attention = tf.split(attention, radix, axis = -1)
        out = tf.keras.layers.Add(name = "{0}split_attention_out".format(prefix))([o * a for o, a in zip(split, attention)])
    else:
        out = tf.keras.layers.Multiply(name = "{0}split_attention_out".format(prefix))([attention, out])
    return out
    
def rsoftmax(x, n_filter, radix, group_size):
    if 1 < radix:
        out = tf.keras.layers.Reshape([group_size, radix, n_filter // group_size])(x)
        out = tf.keras.layers.Permute([2, 1, 3])(out)
        out = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x, axis = 1))(out)
        out = tf.keras.layers.Reshape([1, 1, radix * n_filter])(out)
    else:
        out = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(x)
    return out
    
class rSoftMax(tf.keras.layers.Layer):
    def __init__(self, filters, radix, group_size, **kwargs):
        super(rSoftMax, self).__init__(**kwargs)
        
        self.filters = filters
        self.radix = radix
        self.group_size = group_size
        
        if 1 < radix:
            self.seq1 = tf.keras.layers.Reshape([group_size, radix, filters // group_size])
            self.seq2 = tf.keras.layers.Permute([2, 1, 3])
            self.seq3 = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x, axis = 1))
            self.seq4 = tf.keras.layers.Reshape([1, 1, radix * filters])
            self.seq = [self.seq1, self.seq2, self.seq3, self.seq4]
        else:
            self.seq1 = tf.keras.layers.Activation(tf.keras.activations.sigmoid)
            self.seq = [self.seq1]

    def call(self, inputs):
        out = inputs
        for l in self.seq:
            out = l(out)
        return out
    
    def get_config(self):
        config = super(rSoftMax, self).get_config()
        config["filters"] = self.filters
        config["radix"] = self.radix
        config["group_size"] = self.group_size
        return config

def resnest_block(x, n_filter, stride_size = 1, dilation = 1, group_size = 1, radix = 1, block_width = 64, avd = False, avd_first = False, downsample = None, dropout_rate = 0., expansion = 4, is_first = False, stage = 1, index = 1):
    avd = avd and (1 < stride_size or is_first)
    group_width = int(n_filter * (block_width / 64)) * group_size
    
    out = tf.keras.layers.Conv2D(group_width, 1, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "stage{0}_block{1}_conv1".format(stage, index))(x)
    out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "stage{0}_block{1}_bn1".format(stage, index))(out)
    if 0 < dropout_rate:
        out = tf.keras.layers.Dropout(dropout_rate, name = "stage{0}_block{1}_dropout1".format(stage, index))(out)
    out = tf.keras.layers.Activation("relu", name = "stage{0}_block{1}_act1".format(stage, index))(out)
    
    if avd:
        avd_layer = tf.keras.layers.AveragePooling2D(3, strides = stride_size, padding = "same", name = "stage{0}_block{1}_avd".format(stage, index))
        stride_size = 1
        if avd_first:
            out = avd_layer(out)

    if 0 < radix:
        out = split_attention_block(out, group_width, 3, stride_size, dilation, group_size, radix, dropout_rate, expansion, prefix = "stage{0}_block{1}".format(stage, index))
    else:
        out = tf.keras.layers.Conv2D(group_width, 3, strides = stride_size, dilation_rate = dilation, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "stage{0}_block{1}_conv2".format(stage, index))(out)
        out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "stage{0}_block{1}_bn2".format(stage, index))(out)
        if 0 < dropout_rate:
            out = tf.keras.layers.Dropout(dropout_rate, name = "stage{0}_block{1}_dropout2".format(stage, index))(out)
        out = tf.keras.layers.Activation("relu", name = "stage{0}_block{1}_act2".format(stage, index))(out)
    
    if avd and not avd_first:
        out = avd_layer(out)
    
    out = tf.keras.layers.Conv2D(n_filter * expansion, 1, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "stage{0}_block{1}_conv3".format(stage, index))(out)
    out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "stage{0}_block{1}_bn3".format(stage, index))(out)
    if 0 < dropout_rate:
        out = tf.keras.layers.Dropout(dropout_rate, name = "stage{0}_block{1}_dropout3".format(stage, index))(out)
    residual = x
    if downsample is not None:
        residual = downsample
    out = tf.keras.layers.Add(name = "stage{0}_block{1}_shorcut".format(stage, index))([out, residual])
    out = tf.keras.layers.Activation(tf.keras.activations.relu, name = "stage{0}_block{1}_shorcut_act".format(stage, index))(out)
    return out

def resnest_module(x, n_filter, n_block, stride_size = 1, dilation = 1, group_size = 1, radix = 1, block_width = 64, avg_down = True, avd = False, avd_first = False, dropout_rate = 0., expansion = 4, is_first = True, stage = 1):
    downsample = None
    if stride_size != 1 or tf.keras.backend.int_shape(x)[-1] != (n_filter * expansion):
        if avg_down:
            if dilation == 1:
                downsample = tf.keras.layers.AveragePooling2D(stride_size, strides = stride_size, padding = "same", name = "stage{0}_downsample_avgpool".format(stage))(x)
            else:
                downsample = tf.keras.layers.AveragePooling2D(1, strides = 1, padding = "same", name = "stage{0}_downsample_avgpool".format(stage))(x)
            downsample = tf.keras.layers.Conv2D(n_filter * expansion, 1, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "stage{0}_downsample_conv1".format(stage))(downsample)
            downsample = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "stage{0}_downsample_bn1".format(stage))(downsample)
        else:
            downsample = tf.keras.layers.Conv2D(n_filter * expansion, 1, strides = stride_size, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "stage{0}_downsample_conv1".format(stage))(x)
            downsample = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "stage{0}_downsample_bn1".format(stage))(downsample)
    
    if dilation == 1 or dilation == 2 or dilation == 4:
        out = resnest_block(x, n_filter, stride_size, 2 ** (dilation // 4), group_size, radix, block_width, avd, avd_first, downsample, dropout_rate, expansion, is_first, stage = stage)
    else:
        raise ValueError("unknown dilation size '{0}'".format(dilation))
    
    for index in range(1, n_block):
        out = resnest_block(out, n_filter, 1, dilation, group_size, radix, block_width, avd, avd_first, dropout_rate = dropout_rate, expansion = expansion, stage = stage, index = index + 1)
    return out

def ResNet(x, stack, n_class = 1000, include_top = True, dilation = 1, group_size = 1, radix = 1, block_width = 64, stem_width = 64, deep_stem = False, dilated = False, avg_down = False, avd = False, avd_first = False, dropout_rate = 0., expansion = 4):
    #https://github.com/Burf/ResNeSt-Tensorflow2
    
    #Stem
    if deep_stem:
        out = tf.keras.layers.Conv2D(stem_width, 3, strides = 2, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "stem_conv1")(x)
        out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "stem_bn1")(out)
        out = tf.keras.layers.Activation("relu", name = "stem_act1")(out)
        out = tf.keras.layers.Conv2D(stem_width, 3, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "stem_conv2")(out)
        out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "stem_bn2")(out)
        out = tf.keras.layers.Activation("relu", name = "stem_act2")(out)
        out = tf.keras.layers.Conv2D(stem_width * 2, 3, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "stem_conv3")(out)
        out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "stem_bn3")(out)
        out = tf.keras.layers.Activation("relu", name = "stem_act3")(out)
    else:
        out = tf.keras.layers.Conv2D(64, 7, strides = 2, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "stem_conv1")(x)
        out = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5, name = "stem_bn1")(out)
        out = tf.keras.layers.Activation("relu", name = "stem_act1")(out)
    out = tf.keras.layers.MaxPool2D(3, strides = 2, padding = "same", name = "stem_pooling")(out)
    
    #Stage 1
    out = resnest_module(out, 64, stack[0], 1, 1, group_size, radix, block_width, avg_down, avd, avd_first, expansion = expansion, is_first = False, stage = 1)
    #Stage 2
    out = resnest_module(out, 128, stack[1], 2, 1, group_size, radix, block_width, avg_down, avd, avd_first, expansion = expansion, stage = 2)
    
    if dilated or dilation == 4:
        dilation = [2, 4]
        stride_size = [1, 1]
    elif dilation == 2:
        dilation = [1, 2]
        stride_size = [2, 1]
    else:
        dilation = [1, 1]
        stride_size = [2, 2]
    
    #Stage 3
    out = resnest_module(out, 256, stack[2], stride_size[0], dilation[0], group_size, radix, block_width, avg_down, avd, avd_first, dropout_rate, expansion, stage = 3)
    #Stage 4
    out = resnest_module(out, 512, stack[3], stride_size[1],dilation[1], group_size, radix, block_width, avg_down, avd, avd_first, dropout_rate, expansion, stage = 4)
    
    if include_top:
        out = tf.keras.layers.GlobalAveragePooling2D(name = "feature_avg_pool")(out)
        out = tf.keras.layers.Dense(n_class, activation = tf.keras.activations.softmax, name = "logits")(out)
    return out

_url_format = 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

def load_weight(keras_model, torch_url, group_size = 2):
    """
    https://s3.us-west-1.wasabisys.com/resnest/torch/resnest50-528c19ca.pth > https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest50-528c19ca.pth
    https://s3.us-west-1.wasabisys.com/resnest/torch/resnest101-22405ba7.pth > https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest101-22405ba7.pth
    https://s3.us-west-1.wasabisys.com/resnest/torch/resnest200-75117900.pth > https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest200-75117900.pth
    https://s3.us-west-1.wasabisys.com/resnest/torch/resnest269-0cc87c48.pth > https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest269-0cc87c48.pth
    """
    try:
        import torch
        torch_weight = torch.hub.load_state_dict_from_url(torch_url, map_location = "cpu", progress = True, check_hash = True)
    except:
        print("If you want to use 'resnest weight', please install 'torch 1.1▲'\n{0}".format(traceback.format_exc()))
        return keras_model
    
    weight = {}
    for k, v in dict(torch_weight).items():
        if k.split(".")[-1] in ["weight", "bias", "running_mean", "running_var"]:
            if ("downsample" in k or "conv" in k) and "weight" in k and v.ndim == 4:
                v = v.permute(2, 3, 1, 0)
            elif "fc.weight" in k:
                v = v.t()
            weight[k] = v.cpu().data.numpy()
    
    g = 0
    downsample = []
    keras_weight = []
    for i, (torch_name, torch_weight) in enumerate(weight.items()):
        if i + g < len(keras_model.weights):
            keras_name = keras_model.weights[i + g].name
            if "downsample" in torch_name:
                downsample.append(torch_weight)
                continue
            elif "group" in keras_name:
                g += (group_size - 1)
                torch_weight = tf.split(torch_weight, group_size, axis = -1)
            else:
                torch_weight = [torch_weight]
            keras_weight += torch_weight
    
    for w in keras_model.weights:
        if "downsample" in w.name:
            new_w = downsample.pop(0)
        else:
            new_w = keras_weight.pop(0)
        tf.keras.backend.set_value(w, new_w)
    return keras_model

def resnest50(x, dropout_rate = 0., weights = "imagenet", indices = [0, 1, 2, 3], frozen_stages = -1):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    out = ResNet(x, [3, 4, 6, 3], include_top = False, radix = 2, group_size = 1, block_width = 64, stem_width = 32, deep_stem = True, avg_down = True, avd = True, avd_first = False, dropout_rate = dropout_rate)
    model = tf.keras.Model(x, out)
    
    if weights == "imagenet":
        load_weight(model, resnest_model_urls["resnest50"], group_size = 2 * 1)
    elif weights is not None:
        model.load_weights(weights)
        
    layers = ["stem_pooling", "stage1_block3_shorcut_act", "stage2_block4_shorcut_act", "stage3_block6_shorcut_act", "stage4_block3_shorcut_act"]
    if -1 < frozen_stages:
        for l in model.layers:
            l.trainable = False
            if l.name == layers[frozen_stages]:
                break
    feature = [model.get_layer(l).output for l in layers[1:]]
    
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature
    
def resnest101(x, dropout_rate = 0., weights = "imagenet", indices = [0, 1, 2, 3], frozen_stages = -1):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    out = ResNet(x, [3, 4, 23, 3], include_top = False, radix = 2, group_size = 1, block_width = 64, stem_width = 64, deep_stem = True, avg_down = True, avd = True, avd_first = False, dropout_rate = dropout_rate)
    model = tf.keras.Model(x, out)
    
    if weights == "imagenet":
        load_weight(model, resnest_model_urls["resnest101"], group_size = 2 * 1)
    elif weights is not None:
        model.load_weights(weights)
        
    layers = ["stem_pooling", "stage1_block3_shorcut_act", "stage2_block4_shorcut_act", "stage3_block23_shorcut_act", "stage4_block3_shorcut_act"]
    if -1 < frozen_stages:
        for l in model.layers:
            l.trainable = False
            if l.name == layers[frozen_stages]:
                break
    feature = [model.get_layer(l).output for l in layers[1:]]
    
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def resnest200(x, dropout_rate = 0., weights = "imagenet", indices = [0, 1, 2, 3], frozen_stages = -1):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    out = ResNet(x, [3, 24, 36, 3], include_top = False, radix = 2, group_size = 1, block_width = 64, stem_width = 64, deep_stem = True, avg_down = True, avd = True, avd_first = False, dropout_rate = dropout_rate)
    model = tf.keras.Model(x, out)
    
    if weights == "imagenet":
        load_weight(model, resnest_model_urls["resnest200"], group_size = 2 * 1)
    elif weights is not None:
        model.load_weights(weights)
        
    layers = ["stem_pooling", "stage1_block3_shorcut_act", "stage2_block24_shorcut_act", "stage3_block36_shorcut_act", "stage4_block3_shorcut_act"]
    if -1 < frozen_stages:
        for l in model.layers:
            l.trainable = False
            if l.name == layers[frozen_stages]:
                break
    feature = [model.get_layer(l).output for l in layers[1:]]
    
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def resnest269(x, dropout_rate = 0., weights = "imagenet", indices = [0, 1, 2, 3], frozen_stages = -1):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    out = ResNet(x, [3, 30, 48, 8], include_top = False, radix = 2, group_size = 1, block_width = 64, stem_width = 64, deep_stem = True, avg_down = True, avd = True, avd_first = False, dropout_rate = dropout_rate)
    model = tf.keras.Model(x, out)
    
    if weights == "imagenet":
        load_weight(model, resnest_model_urls["resnest269"], group_size = 2 * 1)
    elif weights is not None:
        model.load_weights(weights)
        
    layers = ["stem_pooling", "stage1_block3_shorcut_act", "stage2_block30_shorcut_act", "stage3_block48_shorcut_act", "stage4_block8_shorcut_act"]
    if -1 < frozen_stages:
        for l in model.layers:
            l.trainable = False
            if l.name == layers[frozen_stages]:
                break
    feature = [model.get_layer(l).output for l in layers[1:]]
    
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature
