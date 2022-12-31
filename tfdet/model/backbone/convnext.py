import tensorflow as tf

def normalize(epsilon = 1e-6, **kwargs):
    return tf.keras.layers.LayerNormalization(epsilon = epsilon, **kwargs)

def convnext_tiny(x, weights = "imagenet", indices = None, normalize = normalize):
    try:
        model = tf.keras.applications.ConvNeXtTiny(input_tensor = x, include_top = False, weights = weights)
    except Exception as e:
        print("If you want to use 'ConvNeXt', please install 'tensorflow 2.11▲'")
        raise e
    
    layers = ["convnext_tiny_downsampling_block_0", "convnext_tiny_downsampling_block_1", "convnext_tiny_downsampling_block_2", model.layers[-1].name]
    feature = []
    for i, l in enumerate(layers):
        out = model.get_layer(l).get_input_at(0)
        out = normalize()(out)
        feature.append(out)
    
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def convnext_small(x, weights = "imagenet", indices = None, normalize = normalize):
    try:
        model = tf.keras.applications.ConvNeXtSmall(input_tensor = x, include_top = False, weights = weights)
    except Exception as e:
        print("If you want to use 'ConvNeXt', please install 'tensorflow 2.11▲'")
        raise e
    
    layers = ["convnext_small_downsampling_block_0", "convnext_small_downsampling_block_1", "convnext_small_downsampling_block_2", model.layers[-1].name]
    feature = []
    for i, l in enumerate(layers):
        out = model.get_layer(l).get_input_at(0)
        out = normalize()(out)
        feature.append(out)
    
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def convnext_base(x, weights = "imagenet", indices = None, normalize = normalize):
    try:
        model = tf.keras.applications.ConvNeXtBase(input_tensor = x, include_top = False, weights = weights)
    except Exception as e:
        print("If you want to use 'ConvNeXt', please install 'tensorflow 2.11▲'")
        raise e
    
    layers = ["convnext_base_downsampling_block_0", "convnext_base_downsampling_block_1", "convnext_base_downsampling_block_2", model.layers[-1].name]
    feature = []
    for i, l in enumerate(layers):
        out = model.get_layer(l).get_input_at(0)
        out = normalize()(out)
        feature.append(out)
    
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def convnext_large(x, weights = "imagenet", indices = None, normalize = normalize):
    try:
        model = tf.keras.applications.ConvNeXtLarge(input_tensor = x, include_top = False, weights = weights)
    except Exception as e:
        print("If you want to use 'ConvNeXt', please install 'tensorflow 2.11▲'")
        raise e
    
    layers = ["convnext_large_downsampling_block_0", "convnext_large_downsampling_block_1", "convnext_large_downsampling_block_2", model.layers[-1].name]
    feature = []
    for i, l in enumerate(layers):
        out = model.get_layer(l).get_input_at(0)
        out = normalize()(out)
        feature.append(out)
    
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def convnext_xlarge(x, weights = "imagenet", indices = None, normalize = normalize):
    try:
        model = tf.keras.applications.ConvNeXtXLarge(input_tensor = x, include_top = False, weights = weights)
    except Exception as e:
        print("If you want to use 'ConvNeXt', please install 'tensorflow 2.11▲'")
        raise e
    
    layers = ["convnext_xlarge_downsampling_block_0", "convnext_xlarge_downsampling_block_1", "convnext_xlarge_downsampling_block_2", model.layers[-1].name]
    feature = []
    for i, l in enumerate(layers):
        out = model.get_layer(l).get_input_at(0)
        out = normalize()(out)
        feature.append(out)
    
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature