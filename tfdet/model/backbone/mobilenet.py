import numpy as np
import tensorflow as tf

def normalize_v2(axis = -1, momentum = 0.9, epsilon = 1e-5, **kwargs):
    return tf.keras.layers.BatchNormalization(axis = axis, momentum = momentum, epsilon = epsilon, **kwargs)

def normalize_v3(axis = -1, momentum = 0.99, epsilon = 1e-3, **kwargs):
    return tf.keras.layers.BatchNormalization(axis = axis, momentum = momentum, epsilon = epsilon, **kwargs)

def _depth(v, divisor = 8, min_value = None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def hard_sigmoid(x):
    return tf.nn.relu6(x + 3.0) * (1.0 / 6.0)

def hard_swish(x):
    return x * hard_sigmoid(x)

def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 2
    if tf.keras.backend.image_data_format() == "channels_last":
        img_dim = 1
    input_size = tf.keras.backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

def _se_block(inputs, filters, se_ratio, prefix):
    x = tf.keras.layers.GlobalAveragePooling2D(keepdims = True, name = prefix + "squeeze_excite/AvgPool")(inputs)
    x = tf.keras.layers.Conv2D(_depth(filters * se_ratio), 1, padding="same", name = prefix + "squeeze_excite/Conv")(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu, name = prefix + "squeeze_excite/Relu")(x)
    x = tf.keras.layers.Conv2D(filters, 1, padding = "same", name = prefix + "squeeze_excite/Conv_1")(x)
    x = tf.keras.layers.Activation(hard_sigmoid, name = prefix + "squeeze_excite/HardSigmoid")(x)
    x = tf.keras.layers.Multiply(name = prefix + "squeeze_excite/Mul")([inputs, x])
    return x

def _inverted_res_block(inputs, filters, kernel_size, stride, expansion, block_id, expand_depth = True, se_ratio = 0.25, normalize = normalize_v3, activation = hard_swish):
    in_channels = tf.keras.backend.int_shape(inputs)[-1]
    x = inputs

    expand_filters = _depth(expansion * in_channels) if expand_depth else int(expansion * in_channels)
    if block_id:
        prefix = f"expanded_conv_{block_id}/"
        x = tf.keras.layers.Conv2D(expand_filters, kernel_size = 1, padding = "same", use_bias = False, name = prefix + "expand")(x)
        x = normalize(name = prefix + "expand/BatchNorm")(x)
        x = tf.keras.layers.Activation(activation, name = prefix + "expand/Activation")(x)
    else:
        prefix = "expanded_conv/"

    # Depthwise 3x3 convolution.
    if stride == 2:
        x = tf.keras.layers.ZeroPadding2D(padding = correct_pad(x, kernel_size), name = prefix + "depthwise/Pad")(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size = kernel_size, strides = stride, use_bias = False, padding = "same" if stride == 1 else "valid", name = prefix + "depthwise")(x)
    x = normalize(name = prefix + "depthwise/BatchNorm")(x)

    x = tf.keras.layers.Activation(activation, name = prefix + "depthwise/Activation")(x)
    if se_ratio is not None:
        x = _se_block(x, expand_filters, se_ratio, prefix)

    # Project with a pointwise 1x1 convolution.
    x = tf.keras.layers.Conv2D(filters, kernel_size = 1, padding = "same", use_bias = False, name = prefix + "project")(x)
    x = normalize(name = prefix + "project/BatchNorm")(x)

    if in_channels == filters and stride == 1:
        return tf.keras.layers.Add(name = prefix + "Add")([inputs, x])
    return x

def MobileNetV2(
    alpha = 1.0,
    include_top = True,
    weights = None,
    input_tensor = None,
    input_shape = None,
    pooling = None,
    normalize = normalize_v2,
    activation = tf.nn.relu6,
    classes = 1000,
    classifier_activation = tf.keras.activations.softmax,
    **kwargs,
):
    #https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v2.py
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor

    first_block_filters = _depth(int(32 * alpha))
    x = tf.keras.layers.Conv2D(first_block_filters, kernel_size = 3, strides = (2, 2), padding = "same", use_bias = False, name = "Stem")(img_input)
    x = normalize(name = "Stem/BatchNorm")(x)
    x = tf.keras.layers.Activation(activation, name = "Stem/Activation")(x)
    
    x = _inverted_res_block(x, filters = _depth(int(16 * alpha)), kernel_size = 3, stride = 1, expansion = 1, block_id = 0, expand_depth = False, se_ratio = None, normalize = normalize, activation = activation)

    x = _inverted_res_block(x, filters = _depth(int(24 * alpha)), kernel_size = 3, stride = 2, expansion = 6, block_id = 1, expand_depth = False, se_ratio = None, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(24 * alpha)), kernel_size = 3, stride = 1, expansion = 6, block_id = 2, expand_depth = False, se_ratio = None, normalize = normalize, activation = activation)

    x = _inverted_res_block(x, filters = _depth(int(32 * alpha)), kernel_size = 3, stride = 2, expansion = 6, block_id = 3, expand_depth = False, se_ratio = None, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(32 * alpha)), kernel_size = 3, stride = 1, expansion = 6, block_id = 4, expand_depth = False, se_ratio = None, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(32 * alpha)), kernel_size = 3, stride = 1, expansion = 6, block_id = 5, expand_depth = False, se_ratio = None, normalize = normalize, activation = activation)

    x = _inverted_res_block(x, filters = _depth(int(64 * alpha)), kernel_size = 3, stride = 2, expansion = 6, block_id = 6, expand_depth = False, se_ratio = None, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(64 * alpha)), kernel_size = 3, stride = 1, expansion = 6, block_id = 7, expand_depth = False, se_ratio = None, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(64 * alpha)), kernel_size = 3, stride = 1, expansion = 6, block_id = 8, expand_depth = False, se_ratio = None, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(64 * alpha)), kernel_size = 3, stride = 1, expansion = 6, block_id = 9, expand_depth = False, se_ratio = None, normalize = normalize, activation = activation)

    x = _inverted_res_block(x, filters = _depth(int(96 * alpha)), kernel_size = 3, stride = 1, expansion = 6, block_id = 10, expand_depth = False, se_ratio = None, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(96 * alpha)), kernel_size = 3, stride = 1, expansion = 6, block_id = 11, expand_depth = False, se_ratio = None, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(96 * alpha)), kernel_size = 3, stride = 1, expansion = 6, block_id = 12, expand_depth = False, se_ratio = None, normalize = normalize, activation = activation)

    x = _inverted_res_block(x, filters = _depth(int(160 * alpha)), kernel_size = 3, stride = 2, expansion = 6, block_id = 13, expand_depth = False, se_ratio = None, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(160 * alpha)), kernel_size = 3, stride = 1, expansion = 6, block_id = 14, expand_depth = False, se_ratio = None, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(160 * alpha)), kernel_size = 3, stride = 1, expansion = 6, block_id = 15, expand_depth = False, se_ratio = None, normalize = normalize, activation = activation)
    
    x = _inverted_res_block(x, filters = _depth(int(320 * alpha)), kernel_size = 3, stride = 1, expansion = 6, block_id = 16, expand_depth = False, se_ratio = None, normalize = normalize, activation = activation)

    last_block_filters = 1280
    if alpha > 1.0:
        last_block_filters = _depth(int(last_block_filters * alpha))

    x = tf.keras.layers.Conv2D(last_block_filters, kernel_size = 1, use_bias = False, name = "Conv")(x)
    x = normalize(name = "Conv/BatchNorm")(x)
    x = tf.keras.layers.Activation(activation, name = "Conv/Activation")(x)

    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(classes, activation = classifier_activation, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), name = "Predictions")(x)

    model = tf.keras.Model(img_input, x)
    if weights is not None:
        model.load_weights(weights)
    return model

def MobileNetV3Small(
    alpha = 1.0,
    include_top = True,
    weights = None,
    input_tensor = None,
    input_shape = None,
    pooling = None,
    se_ratio = 0.25,
    normalize = normalize_v3,
    activation = hard_swish,
    classes = 1000,
    classifier_activation = tf.keras.activations.softmax,
    **kwargs,
):
    #https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v3.py
    #https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor

    x = tf.keras.layers.Conv2D(16, kernel_size = 3, strides = (2, 2), padding = "same", use_bias = False, name = "Stem")(img_input)
    x = normalize(name = "Stem/BatchNorm")(x)
    x = tf.keras.layers.Activation(activation, name = "Stem/Activation")(x)
    
    x = _inverted_res_block(x, filters = _depth(int(16 * alpha)), kernel_size = 3, stride = 2, expansion = 1, block_id = 0, se_ratio = se_ratio, normalize = normalize, activation = tf.keras.activations.relu)
    
    x = _inverted_res_block(x, filters = _depth(int(24 * alpha)), kernel_size = 3, stride = 2, expansion = 72 / 16, block_id = 1, se_ratio = None, normalize = normalize, activation = tf.keras.activations.relu)
    x = _inverted_res_block(x, filters = _depth(int(24 * alpha)), kernel_size = 3, stride = 1, expansion = 88 / 24, block_id = 2, se_ratio = None, normalize = normalize, activation = tf.keras.activations.relu)
    
    x = _inverted_res_block(x, filters = _depth(int(40 * alpha)), kernel_size = 5, stride = 2, expansion = 4, block_id = 3, se_ratio = se_ratio, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(40 * alpha)), kernel_size = 5, stride = 1, expansion = 6, block_id = 4, se_ratio = se_ratio, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(40 * alpha)), kernel_size = 5, stride = 1, expansion = 6, block_id = 5, se_ratio = se_ratio, normalize = normalize, activation = activation)
    
    x = _inverted_res_block(x, filters = _depth(int(48 * alpha)), kernel_size = 5, stride = 1, expansion = 3, block_id = 6, se_ratio = se_ratio, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(48 * alpha)), kernel_size = 5, stride = 1, expansion = 3, block_id = 7, se_ratio = se_ratio, normalize = normalize, activation = activation)
    
    x = _inverted_res_block(x, filters = _depth(int(96 * alpha)), kernel_size = 5, stride = 2, expansion = 6, block_id = 8, se_ratio = se_ratio, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(96 * alpha)), kernel_size = 5, stride = 1, expansion = 6, block_id = 9, se_ratio = se_ratio, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(96 * alpha)), kernel_size = 5, stride = 1, expansion = 6, block_id = 10, se_ratio = se_ratio, normalize = normalize, activation = activation)
    
    last_conv_ch = _depth(tf.keras.backend.int_shape(x)[-1] * 6)
    last_block_filters = 1024
    if alpha > 1.0:
        last_block_filters = _depth(int(last_block_filters * alpha))

    x = tf.keras.layers.Conv2D(last_conv_ch, kernel_size = 1, use_bias = False, name = "Conv")(x)
    x = normalize(name = "Conv/BatchNorm")(x)
    x = tf.keras.layers.Activation(activation, name = "Conv/Activation")(x)

    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(last_block_filters, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), name = "Dense")(x)
        x = tf.keras.layers.Dense(classes, activation = classifier_activation, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), name = "Predictions")(x)

    model = tf.keras.Model(img_input, x)
    if weights is not None:
        model.load_weights(weights)
    return model

def MobileNetV3Large(
    alpha = 1.0,
    include_top = True,
    weights = None,
    input_tensor = None,
    input_shape = None,
    pooling = None,
    se_ratio = 0.25,
    normalize = normalize_v3,
    activation = hard_swish,
    classes = 1000,
    classifier_activation = tf.keras.activations.softmax,
    **kwargs,
):
    #https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v3.py
    #https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor

    x = tf.keras.layers.Conv2D(16, kernel_size = 3, strides = (2, 2), padding = "same", use_bias = False, name = "Stem")(img_input)
    x = normalize(name = "Stem/BatchNorm")(x)
    x = tf.keras.layers.Activation(activation, name = "Stem/Activation")(x)
    
    x = _inverted_res_block(x, filters = _depth(int(16 * alpha)), kernel_size = 3, stride = 1, expansion = 1, block_id = 0, se_ratio = None, normalize = normalize, activation = tf.keras.activations.relu)
    
    x = _inverted_res_block(x, filters = _depth(int(24 * alpha)), kernel_size = 3, stride = 2, expansion = 4, block_id = 1, se_ratio = None, normalize = normalize, activation = tf.keras.activations.relu)
    x = _inverted_res_block(x, filters = _depth(int(24 * alpha)), kernel_size = 3, stride = 1, expansion = 3, block_id = 2, se_ratio = None, normalize = normalize, activation = tf.keras.activations.relu)
    
    x = _inverted_res_block(x, filters = _depth(int(40 * alpha)), kernel_size = 5, stride = 2, expansion = 3, block_id = 3, se_ratio = se_ratio, normalize = normalize, activation = tf.keras.activations.relu)
    x = _inverted_res_block(x, filters = _depth(int(40 * alpha)), kernel_size = 5, stride = 1, expansion = 3, block_id = 4, se_ratio = se_ratio, normalize = normalize, activation = tf.keras.activations.relu)
    x = _inverted_res_block(x, filters = _depth(int(40 * alpha)), kernel_size = 5, stride = 1, expansion = 3, block_id = 5, se_ratio = se_ratio, normalize = normalize, activation = tf.keras.activations.relu)
    
    x = _inverted_res_block(x, filters = _depth(int(80 * alpha)), kernel_size = 3, stride = 2, expansion = 6, block_id = 6, se_ratio = None, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(80 * alpha)), kernel_size = 3, stride = 1, expansion = 2.5, block_id = 7, se_ratio = None, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(80 * alpha)), kernel_size = 3, stride = 1, expansion = 2.3, block_id = 8, se_ratio = None, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(80 * alpha)), kernel_size = 3, stride = 1, expansion = 2.3, block_id = 9, se_ratio = None, normalize = normalize, activation = activation)
    
    x = _inverted_res_block(x, filters = _depth(int(112 * alpha)), kernel_size = 3, stride = 1, expansion = 6, block_id = 10, se_ratio = se_ratio, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(112 * alpha)), kernel_size = 3, stride = 1, expansion = 6, block_id = 11, se_ratio = se_ratio, normalize = normalize, activation = activation)
    
    x = _inverted_res_block(x, filters = _depth(int(160 * alpha)), kernel_size = 5, stride = 2, expansion = 6, block_id = 12, se_ratio = se_ratio, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(160 * alpha)), kernel_size = 5, stride = 1, expansion = 6, block_id = 13, se_ratio = se_ratio, normalize = normalize, activation = activation)
    x = _inverted_res_block(x, filters = _depth(int(160 * alpha)), kernel_size = 5, stride = 1, expansion = 6, block_id = 14, se_ratio = se_ratio, normalize = normalize, activation = activation)
    
    last_conv_ch = _depth(tf.keras.backend.int_shape(x)[-1] * 6)
    last_block_filters = 1280
    if alpha > 1.0:
        last_block_filters = _depth(int(last_block_filters * alpha))

    x = tf.keras.layers.Conv2D(last_conv_ch, kernel_size = 1, use_bias = False, name = "Conv")(x)
    x = normalize(name = "Conv/BatchNorm")(x)
    x = tf.keras.layers.Activation(activation, name = "Conv/Activation")(x)

    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(last_block_filters, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), name = "Dense")(x)
        x = tf.keras.layers.Dense(classes, activation = classifier_activation, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), name = "Predictions")(x)

    model = tf.keras.Model(img_input, x)
    if weights is not None:
        model.load_weights(weights)
    return model

mobilenet_urls = {
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
    "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth"
}

mobilenet_imagenet_v2_urls = {
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth",
    "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth",
}

def load_weight(keras_model, torch_url):
    try:
        import torch
        torch_weight = torch.hub.load_state_dict_from_url(torch_url, map_location = "cpu", progress = True, check_hash = True)
    except:
        print("If you want to use 'mobilenet weight', please install 'torch 1.1▲'\n{0}".format(traceback.format_exc()))
        return keras_model
    
    conv_flag = False
    conv = []
    bn = {"weight":[], "bias":[], "running_mean":[], "running_var":[]}
    fc = []
    for k, v in dict(torch_weight).items():
        if k.split(".")[-1] in ["weight", "bias", "running_mean", "running_var"]:
            if conv_flag and "weight" in k:
                conv_flag = False
            if "weight" in k and v.ndim == 4 or "bias" in k and conv_flag:
                if v.ndim == 4:
                    v = v.permute(2, 3, 1, 0)
                    conv_flag = True
                else:
                    conv_flag = False
                conv.append(v.cpu().data.numpy())
            elif "classifier" in k:
                if "weight" in k:
                    v = v.t()
                fc.append(v.cpu().data.numpy())
            else: #bn
                bn[k.split(".")[-1]].append(v.cpu().data.numpy())
    bn = [b for a in [[w, b, m, v] for w, b, m, v in zip(*list(bn.values()))] for b in a]
    
    for w in keras_model.weights:
        if "BatchNorm" in w.name:
            new_w = bn.pop(0)
        elif "Predictions" in w.name or "Dense" in w.name:
            new_w = fc.pop(0)
        else:
            new_w = conv.pop(0)
            if tf.keras.backend.int_shape(w) != new_w.shape: #depthwise
                new_w = np.transpose(new_w, [0, 1, 3, 2])
        tf.keras.backend.set_value(w, new_w)
    return keras_model

def mobilenet_v2(x, weights = "imagenet_v2", indices = [1, 2, 4, 7], frozen_stages = -1):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    model = MobileNetV2(input_tensor = x, include_top = False, weights = None)
    if weights == "imagenet":
        load_weight(model, mobilenet_urls["mobilenet_v2"])
    elif weights == "imagenet_v2":
        load_weight(model, mobilenet_imagenet_v2_urls["mobilenet_v2"])
    elif weights is not None:
        model.load_weights(weights)
    
    layers = ["Stem/Activation", "expanded_conv/project/BatchNorm", "expanded_conv_2/Add", "expanded_conv_5/Add", "expanded_conv_9/Add", "expanded_conv_12/Add", "expanded_conv_15/Add", "expanded_conv_16/project/BatchNorm", "Conv/Activation"]
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

def mobilenet_v3_small(x, weights = "imagenet", indices = [0, 1, 3, 5], frozen_stages = -1):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    model = MobileNetV3Small(input_tensor = x, include_top = False, weights = None)
    if weights == "imagenet":
        load_weight(model, mobilenet_urls["mobilenet_v3_small"])
    elif weights is not None:
        model.load_weights(weights)
    
    layers = ["Stem/Activation", "expanded_conv/project/BatchNorm", "expanded_conv_2/Add", "expanded_conv_5/Add", "expanded_conv_7/Add", "expanded_conv_10/Add", "Conv/Activation"]
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

def mobilenet_v3_large(x, weights = "imagenet_v2", indices = [1, 2, 4, 6], frozen_stages = -1):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    model = MobileNetV3Large(input_tensor = x, include_top = False, weights = None)
    if weights == "imagenet":
        load_weight(model, mobilenet_urls["mobilenet_v3_large"])
    elif weights == "imagenet_v2":
        load_weight(model, mobilenet_imagenet_v2_urls["mobilenet_v3_large"])
    elif weights is not None:
        model.load_weights(weights)
    
    layers = ["Stem/Activation", "expanded_conv/Add", "expanded_conv_2/Add", "expanded_conv_5/Add", "expanded_conv_9/Add", "expanded_conv_11/Add", "expanded_conv_14/Add", "Conv/Activation"]
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