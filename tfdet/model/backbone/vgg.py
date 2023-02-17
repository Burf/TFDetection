import tensorflow as tf

from tfdet.core.ops import AdaptiveAveragePooling2D

def normalize(axis = -1, momentum = 0.9, epsilon = 1e-5, **kwargs):
    return tf.keras.layers.BatchNormalization(axis = axis, momentum = momentum, epsilon = epsilon, **kwargs)

def VGG(blocks = [2, 2, 3, 3, 3],
        include_top = True,
        weights = None,
        input_tensor = None,
        input_shape = None,
        normalize = None,
        activation = tf.keras.activations.relu,
        pooling = None,
        classes = 1000,
        classifier_activation = tf.keras.activations.softmax,
    ):
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor
            
    out = img_input
    for i, (block, feature) in enumerate(zip(blocks, [64, 128,256, 512, 512])):
        for j in range(block):
            out = tf.keras.layers.Conv2D(feature, 3, padding = "same", name = "block{0}_conv{1}".format(i + 1, j + 1))(out)
            if normalize is not None:
                out = normalize(name = "block{0}_norm{1}".format(i + 1, j + 1))(out)
            out = tf.keras.layers.Activation(activation, name = "block{0}_act{1}".format(i + 1, j + 1))(out)
        out = tf.keras.layers.MaxPooling2D(2, strides = 2, name = "block{0}_pool".format(i + 1))(out)

    if include_top:
        out = AdaptiveAveragePooling2D([7, 7], method = "bilinear", name = "adaptive_avg_pool")(out)
        out = tf.keras.layers.Reshape([7 * 7 * 512], name = "flatten")(out)
        out = tf.keras.layers.Dense(4096, activation = activation, name = "fc1")(out)
        out = tf.keras.layers.Dense(4096, activation = activation, name = "fc2")(out)
        out = tf.keras.layers.Dense(classes, activation = classifier_activation, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), name="predictions")(out)

    model = tf.keras.Model(img_input, out)
    if weights is not None:
        model.load_weights(weights)
    return model

vgg_urls = {
    "vgg11":"https://download.pytorch.org/models/vgg11-8a719046.pth",
    "vgg11_bn":"https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13":"https://download.pytorch.org/models/vgg13-19584684.pth",
    "vgg13_bn":"https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16":"https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg16_bn":"https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19":"https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg19_bn":"https://download.pytorch.org/models/vgg19_bn-c79401a0.pth"
}

def load_weight(keras_model, torch_url):
    try:
        import torch
        torch_weight = torch.hub.load_state_dict_from_url(torch_url, map_location = "cpu", progress = True, check_hash = True)
    except:
        print("If you want to use 'vgg weight', please install 'torch 1.1▲'\n{0}".format(traceback.format_exc()))
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
        if "norm" in w.name:
            new_w = bn.pop(0)
        elif "conv" in w.name:
            new_w = conv.pop(0)
        else:
            new_w = fc.pop(0)
        tf.keras.backend.set_value(w, new_w)
    return keras_model

def vgg11(x, weights = "imagenet", indices = [1, 2, 3, 4], frozen_stages = -1):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    model = VGG([1, 1, 2, 2, 2], input_tensor = x, include_top = False, weights = None)
    if weights == "imagenet":
        load_weight(model, vgg_urls["vgg11"])
    elif weights is not None:
        model.load_weights(weights)
    
    layers = [None, "block1_pool", "block2_pool", "block3_pool", "block4_pool", "block5_pool"]
    if 0 < frozen_stages:
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

def vgg11_bn(x, weights = "imagenet", indices = [1, 2, 3, 4], frozen_stages = -1):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    model = VGG([1, 1, 2, 2, 2], input_tensor = x, normalize = normalize, include_top = False, weights = None)
    if weights == "imagenet":
        load_weight(model, vgg_urls["vgg11_bn"])
    elif weights is not None:
        model.load_weights(weights)
    
    layers = [None, "block1_pool", "block2_pool", "block3_pool", "block4_pool", "block5_pool"]
    if 0 < frozen_stages:
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

def vgg13(x, weights = "imagenet", indices = [1, 2, 3, 4], frozen_stages = -1):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    model = VGG([2, 2, 2, 2, 2], input_tensor = x, include_top = False, weights = None)
    if weights == "imagenet":
        load_weight(model, vgg_urls["vgg13"])
    elif weights is not None:
        model.load_weights(weights)
    
    layers = [None, "block1_pool", "block2_pool", "block3_pool", "block4_pool", "block5_pool"]
    if 0 < frozen_stages:
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

def vgg13_bn(x, weights = "imagenet", indices = [1, 2, 3, 4], frozen_stages = -1):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    model = VGG([2, 2, 2, 2, 2], input_tensor = x, normalize = normalize, include_top = False, weights = None)
    if weights == "imagenet":
        load_weight(model, vgg_urls["vgg13_bn"])
    elif weights is not None:
        model.load_weights(weights)
    
    layers = [None, "block1_pool", "block2_pool", "block3_pool", "block4_pool", "block5_pool"]
    if 0 < frozen_stages:
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

def vgg16(x, weights = "imagenet", indices = [1, 2, 3, 4], frozen_stages = -1):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    model = VGG([2, 2, 3, 3, 3], input_tensor = x, include_top = False, weights = None)
    if weights == "imagenet":
        load_weight(model, vgg_urls["vgg16"])
    elif weights is not None:
        model.load_weights(weights)
    
    layers = [None, "block1_pool", "block2_pool", "block3_pool", "block4_pool", "block5_pool"]
    if 0 < frozen_stages:
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

def vgg16_bn(x, weights = "imagenet", indices = [1, 2, 3, 4], frozen_stages = -1):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    model = VGG([2, 2, 3, 3, 3], input_tensor = x, normalize = normalize, include_top = False, weights = None)
    if weights == "imagenet":
        load_weight(model, vgg_urls["vgg16_bn"])
    elif weights is not None:
        model.load_weights(weights)
    
    layers = [None, "block1_pool", "block2_pool", "block3_pool", "block4_pool", "block5_pool"]
    if 0 < frozen_stages:
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
    
def vgg19(x, weights = "imagenet", indices = [1, 2, 3, 4], frozen_stages = -1):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    model = VGG([2, 2, 4, 4, 4], input_tensor = x, normalize = normalize, include_top = False, weights = None)
    if weights == "imagenet":
        load_weight(model, vgg_urls["vgg19"])
    elif weights is not None:
        model.load_weights(weights)
    
    layers = [None, "block1_pool", "block2_pool", "block3_pool", "block4_pool", "block5_pool"]
    if 0 < frozen_stages:
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
    return 
    
def vgg19_bn(x, weights = "imagenet", indices = [1, 2, 3, 4], frozen_stages = -1):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    model = VGG([2, 2, 4, 4, 4], input_tensor = x, normalize = normalize, include_top = False, weights = None)
    if weights == "imagenet":
        load_weight(model, vgg_urls["vgg19_bn"])
    elif weights is not None:
        model.load_weights(weights)
    
    layers = [None, "block1_pool", "block2_pool", "block3_pool", "block4_pool", "block5_pool"]
    if 0 < frozen_stages:
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