import tensorflow as tf

def mobilenet(x, weights = "imagenet", indices = None):
    model = tf.keras.applications.MobileNet(input_tensor = x, include_top = False, weights = weights)
    layers = ["conv_pw_3_relu", "conv_pw_5_relu", "conv_pw_11_relu", "conv_pw_13_relu"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature
    
def mobilenet_v2(x, weights = "imagenet", indices = None):
    model = tf.keras.applications.MobileNetV2(input_tensor = x, include_top = False, weights = weights)
    layers = ["block_2_add", "block_5_add", "block_12_add", "block_16_project_BN"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def mobilenet_v3_small(x, weights = "imagenet", indices = None):
    try:
        model = tf.keras.applications.MobileNetV3Small(input_tensor = x, include_top = False, include_preprocessing = False, weights = weights)
    except Exception as e:
        print("If you want to use 'MobileNetV3', please install 'tensorflow 2.6▲'")
        raise e
    layers = ["expanded_conv/project/BatchNorm", "expanded_conv_2/Add", "expanded_conv_7/Add", "expanded_conv_10/Add"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def mobilenet_v3_large(x, weights = "imagenet", indices = None):
    try:
        model = tf.keras.applications.MobileNetV3Large(input_tensor = x, include_top = False, include_preprocessing = False, weights = weights)
    except Exception as e:
        print("If you want to use 'MobileNetV3', please install 'tensorflow 2.6▲'")
        raise e
    layers = ["expanded_conv_2/Add", "expanded_conv_5/Add", "expanded_conv_11/Add", "expanded_conv_14/Add"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature