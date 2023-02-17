import tensorflow as tf

def conv_block(x, growth_rate, name):
    x1 = tf.keras.layers.BatchNormalization(momentum = 0.9, epsilon = 1e-5, name = name + "_0_bn")(x)
    x1 = tf.keras.layers.Activation("relu", name = name + "_0_relu")(x1)
    x1 = tf.keras.layers.Conv2D(4 * growth_rate, 1, use_bias=False, name = name + "_1_conv")(x1)
    x1 = tf.keras.layers.BatchNormalization(momentum = 0.9, epsilon = 1e-5, name = name + "_1_bn")(x1)
    x1 = tf.keras.layers.Activation("relu", name = name + "_1_relu")(x1)
    x1 = tf.keras.layers.Conv2D(growth_rate, 3, padding="same", use_bias=False, name = name + "_2_conv")(x1)
    x = tf.keras.layers.Concatenate(name = name + "_concat")([x, x1])
    return x

def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 32, name = name + "_block" + str(i + 1))
    return x

def transition_block(x, reduction, name):
    x = tf.keras.layers.BatchNormalization(momentum = 0.9, epsilon = 1e-5, name = name + "_bn")(x)
    x = tf.keras.layers.Activation("relu", name = name + "_relu")(x)
    x = tf.keras.layers.Conv2D(int(tf.keras.backend.int_shape(x)[-1] * reduction), 1, use_bias=False, name = name + "_conv")(x)
    x = tf.keras.layers.AveragePooling2D(2, strides = 2, name = name + "_pool")(x)
    return x

def DenseNet(
    blocks,
    include_top = True,
    weights = None,
    input_tensor = None,
    input_shape = None,
    pooling = None,
    classes = 1000,
    classifier_activation = tf.keras.activations.softmax,
):
    #https://github.com/keras-team/keras/blob/master/keras/applications/densenet.py
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor

    x = tf.keras.layers.ZeroPadding2D(padding = ((3, 3), (3, 3)))(img_input)
    x = tf.keras.layers.Conv2D(64, 7, strides = 2, use_bias = False, name = "conv1/conv")(x)
    x = tf.keras.layers.BatchNormalization(momentum = 0.9, epsilon = 1e-5, name = "conv1/bn")(x)
    x = tf.keras.layers.Activation("relu", name = "conv1/relu")(x)
    x = tf.keras.layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(x)
    x = tf.keras.layers.MaxPooling2D(3, strides = 2, name = "pool1")(x)

    x = dense_block(x, blocks[0], name = "conv2")
    x = transition_block(x, 0.5, name = "pool2")
    x = dense_block(x, blocks[1], name = "conv3")
    x = transition_block(x, 0.5, name = "pool3")
    x = dense_block(x, blocks[2], name = "conv4")
    x = transition_block(x, 0.5, name = "pool4")
    x = dense_block(x, blocks[3], name = "conv5")

    x = tf.keras.layers.BatchNormalization(momentum = 0.9, epsilon = 1e-5, name = "bn")(x)
    x = tf.keras.layers.Activation("relu", name = "relu")(x)

    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D(name = "avg_pool")(x)
        x = tf.keras.layers.Dense(classes, activation = classifier_activation, name = "predictions")(x)

    model = tf.keras.Model(img_input, x)
    if weights is not None:
        model.load_weights(weights)
    return model

densenet_urls = {
    "densenet121":"https://download.pytorch.org/models/densenet121-a639ec97.pth",
    "densenet169":"https://download.pytorch.org/models/densenet169-b2777c0a.pth",
    "densenet201":"https://download.pytorch.org/models/densenet201-c1103571.pth"
}

def load_weight(keras_model, torch_url):
    try:
        import torch
        torch_weight = torch.hub.load_state_dict_from_url(torch_url, map_location = "cpu", progress = True, check_hash = True)
    except:
        print("If you want to use 'densenet weight', please install 'torch 1.1▲'\n{0}".format(traceback.format_exc()))
        return keras_model
    
    conv = []
    bn = {"weight":[], "bias":[], "running_mean":[], "running_var":[]}
    fc = []
    for k, v in dict(torch_weight).items():
        if k.split(".")[-1] in ["weight", "bias", "running_mean", "running_var"]:
            if "conv" in k and "weight" in k and v.ndim == 4:
                conv.append(v.permute(2, 3, 1, 0).cpu().data.numpy())
            elif "classifier" in k:
                if "weight" in k:
                    v = v.t()
                fc.append(v.cpu().data.numpy())
            else: #bn
                bn[k.split(".")[-1]].append(v.cpu().data.numpy())
    bn = [b for a in [[w, b, m, v] for w, b, m, v in zip(*list(bn.values()))] for b in a]
    
    for w in keras_model.weights:
        if "bn" in w.name:
            new_w = bn.pop(0)
        elif "predictions" in w.name:
            new_w = fc.pop(0)
        else:
            new_w = conv.pop(0)
        tf.keras.backend.set_value(w, new_w)
    return keras_model

def densenet121(x, weights = "imagenet", indices = [0, 1, 2, 3], frozen_stages = -1):
    model = DenseNet([6, 12, 24, 16], input_tensor = x, include_top = False, weights = None)
    if weights == "imagenet":
        load_weight(model, densenet_urls["densenet121"])
    elif weights is not None:
        model.load_weights(weights)
    
    layers = ["conv1/relu", "conv2_block6_concat", "conv3_block12_concat", "conv4_block24_concat", "conv5_block16_concat"]
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

def densenet169(x, weights = "imagenet", indices = [0, 1, 2, 3], frozen_stages = -1):
    model = DenseNet([6, 12, 32, 32], input_tensor = x, include_top = False, weights = None)
    if weights == "imagenet":
        load_weight(model, densenet_urls["densenet169"])
    elif weights is not None:
        model.load_weights(weights)
    
    layers = ["conv1/relu", "conv2_block6_concat", "conv3_block12_concat", "conv4_block32_concat", "conv5_block32_concat"]
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

def densenet201(x, weights = "imagenet", indices = [0, 1, 2, 3], frozen_stages = -1):
    model = DenseNet([6, 12, 48, 32], input_tensor = x, include_top = False, weights = None)
    if weights == "imagenet":
        load_weight(model, densenet_urls["densenet201"])
    elif weights is not None:
        model.load_weights(weights)
    
    layers = ["conv1/relu", "conv2_block6_concat", "conv3_block12_concat", "conv4_block48_concat", "conv5_block32_concat"]
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