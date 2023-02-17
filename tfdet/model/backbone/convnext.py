import tensorflow as tf

def normalize(epsilon = 1e-6, **kwargs):
    return tf.keras.layers.LayerNormalization(epsilon = epsilon, **kwargs)

def convnext_tiny(x, weights = "imagenet", indices = [0, 1, 2, 3], frozen_stages = -1, normalize = normalize):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    try:
        tf.keras.applications.ConvNeXtTiny
    except Exception as e:
        print("If you want to use 'ConvNeXt', please install 'tensorflow 2.10▲'")
        raise e
    model = tf.keras.applications.ConvNeXtTiny(input_tensor = x, include_top = False, include_preprocessing = False, weights = weights)
    
    layers = ["convnext_tiny_stem"]
    target = ["convnext_tiny_downsampling_block_0", "convnext_tiny_downsampling_block_1", "convnext_tiny_downsampling_block_2", model.layers[-1].name]
    for name in target:
        out = model.get_layer(name).get_input_at(0)
        layers.append(out.name.split("/")[0])
    
    if -1 < frozen_stages:
        for l in model.layers:
            l.trainable = False
            if l.name == layers[frozen_stages]:
                break
    feature = []
    for i, l in enumerate(layers[1:]):
        out = model.get_layer(l).output
        out = normalize()(out)
        feature.append(out)
    
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def convnext_small(x, weights = "imagenet", indices = [0, 1, 2, 3], frozen_stages = -1, normalize = normalize):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    try:
        tf.keras.applications.ConvNeXtSmall
    except Exception as e:
        print("If you want to use 'ConvNeXt', please install 'tensorflow 2.10▲'")
        raise e
    model = tf.keras.applications.ConvNeXtSmall(input_tensor = x, include_top = False, include_preprocessing = False, weights = weights)
    
    layers = ["convnext_small_stem"]
    target =["convnext_small_downsampling_block_0", "convnext_small_downsampling_block_1", "convnext_small_downsampling_block_2", model.layers[-1].name]
    for name in target:
        out = model.get_layer(name).get_input_at(0)
        layers.append(out.name.split("/")[0])
    
    if -1 < frozen_stages:
        for l in model.layers:
            l.trainable = False
            if l.name == layers[frozen_stages]:
                break
    feature = []
    for i, l in enumerate(layers[1:]):
        out = model.get_layer(l).output
        out = normalize()(out)
        feature.append(out)
    
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def convnext_base(x, weights = "imagenet", indices = [0, 1, 2, 3], frozen_stages = -1, normalize = normalize):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    try:
        tf.keras.applications.ConvNeXtBase
    except Exception as e:
        print("If you want to use 'ConvNeXt', please install 'tensorflow 2.10▲'")
        raise e
    model = tf.keras.applications.ConvNeXtBase(input_tensor = x, include_top = False, include_preprocessing = False, weights = weights)
    
    layers = ["convnext_base_stem"]
    target = ["convnext_base_downsampling_block_0", "convnext_base_downsampling_block_1", "convnext_base_downsampling_block_2", model.layers[-1].name]
    for name in target:
        out = model.get_layer(name).get_input_at(0)
        layers.append(out.name.split("/")[0])
    
    if -1 < frozen_stages:
        for l in model.layers:
            l.trainable = False
            if l.name == layers[frozen_stages]:
                break
    feature = []
    for i, l in enumerate(layers[1:]):
        out = model.get_layer(l).output
        out = normalize()(out)
        feature.append(out)
    
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def convnext_large(x, weights = "imagenet", indices = [0, 1, 2, 3], frozen_stages = -1, normalize = normalize):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    try:
        tf.keras.applications.ConvNeXtLarge
    except Exception as e:
        print("If you want to use 'ConvNeXt', please install 'tensorflow 2.10▲'")
        raise e
    model = tf.keras.applications.ConvNeXtLarge(input_tensor = x, include_top = False, include_preprocessing = False, weights = weights)
    
    layers = ["convnext_large_stem"]
    target = ["convnext_large_downsampling_block_0", "convnext_large_downsampling_block_1", "convnext_large_downsampling_block_2", model.layers[-1].name]
    for name in target:
        out = model.get_layer(name).get_input_at(0)
        layers.append(out.name.split("/")[0])
    
    if -1 < frozen_stages:
        for l in model.layers:
            l.trainable = False
            if l.name == layers[frozen_stages]:
                break
    feature = []
    for i, l in enumerate(layers[1:]):
        out = model.get_layer(l).output
        out = normalize()(out)
        feature.append(out)
    
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def convnext_xlarge(x, weights = "imagenet", indices = [0, 1, 2, 3], frozen_stages = -1, normalize = normalize):
    """
    imagenet > normalize(x, rmean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    """
    try:
        tf.keras.applications.ConvNeXtXLarge
    except Exception as e:
        print("If you want to use 'ConvNeXt', please install 'tensorflow 2.10▲'")
        raise e
    model = tf.keras.applications.ConvNeXtXLarge(input_tensor = x, include_top = False, include_preprocessing = False, weights = weights)
    
    layers = ["convnext_xlarge_stem"]
    target = ["convnext_xlarge_downsampling_block_0", "convnext_xlarge_downsampling_block_1", "convnext_xlarge_downsampling_block_2", model.layers[-1].name]
    for name in target:
        out = model.get_layer(name).get_input_at(0)
        layers.append(out.name.split("/")[0])
    
    if -1 < frozen_stages:
        for l in model.layers:
            l.trainable = False
            if l.name == layers[frozen_stages]:
                break
    feature = []
    for i, l in enumerate(layers[1:]):
        out = model.get_layer(l).output
        out = normalize()(out)
        feature.append(out)
    
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature