import tensorflow as tf

def densenet121(x, weights = "imagenet", indices = None):
    model = tf.keras.applications.DenseNet121(input_tensor = x, include_top = False, weights = weights)
    layers = ["conv2_block6_concat", "conv3_block12_concat", "conv4_block24_concat", "conv5_block16_concat"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def densenet169(x, weights = "imagenet", indices = None):
    model = tf.keras.applications.DenseNet169(input_tensor = x, include_top = False, weights = weights)
    layers = ["conv2_block6_concat", "conv3_block12_concat", "conv4_block32_concat", "conv5_block32_concat"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def densenet201(x, weights = "imagenet", indices = None):
    model = tf.keras.applications.DenseNet201(input_tensor = x, include_top = False, weights = weights)
    layers = ["conv2_block6_concat", "conv3_block12_concat", "conv4_block48_concat", "conv5_block32_concat"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature