import tensorflow as tf

def effnet_b0(x, weights = "imagenet", indices = None):
    model = tf.keras.applications.EfficientNetB0(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2b_add", "block3b_add", "block5c_add", "block7a_project_bn"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def effnet_b1(x, weights = "imagenet", indices = None):
    model = tf.keras.applications.EfficientNetB1(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2c_add", "block3c_add", "block5d_add", "block7b_add"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def effnet_b2(x, weights = "imagenet", indices = None):
    model = tf.keras.applications.EfficientNetB2(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2c_add", "block3c_add", "block5d_add", "block7b_add"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def effnet_b3(x, weights = "imagenet", indices = None):
    model = tf.keras.applications.EfficientNetB3(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2c_add", "block3c_add", "block5e_add", "block7b_add"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def effnet_b4(x, weights = "imagenet", indices = None):
    model = tf.keras.applications.EfficientNetB4(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2d_add", "block3d_add", "block5f_add", "block7b_add"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def effnet_b5(x, weights = "imagenet", indices = None):
    model = tf.keras.applications.EfficientNetB5(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2e_add", "block3e_add", "block5g_add", "block7c_add"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def effnet_b6(x, weights = "imagenet", indices = None):
    model = tf.keras.applications.EfficientNetB6(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2f_add", "block3f_add", "block5h_add", "block7c_add"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def effnet_b7(x, weights = "imagenet", indices = None):
    model = tf.keras.applications.EfficientNetB7(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2g_add", "block3g_add", "block5j_add", "block7d_add"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

try:
    #https://github.com/Burf/EfficientNet-Lite-Tensorflow2
    import keras
    
    def efficientnet_lite(width_coefficient,
                          depth_coefficient,
                          default_size,
                          dropout_rate=0.2,
                          drop_connect_rate=0.2,
                          depth_divisor=8,
                          activation_fn=tf.nn.relu6,
                          blocks_args=[{k:v if "se_ratio" not in k else 0. for k, v in arg.items()} for arg in keras.applications.efficientnet.DEFAULT_BLOCKS_ARGS],
                          block = keras.applications.efficientnet.block,
                          conv_kernel_initializer = keras.applications.efficientnet.CONV_KERNEL_INITIALIZER,
                          dense_kernel_initializer = keras.applications.efficientnet.DENSE_KERNEL_INITIALIZER,
                          include_top=True,
                          input_tensor=None,
                          input_shape=None,
                          pooling=None,
                          classes=1000,
                          weights = None,
                          **kwargs):
        #https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
        #https://github.com/Burf/EfficientNet-Lite-Tensorflow2
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
        
        def round_filters(filters, divisor=depth_divisor):
            """Round number of filters based on depth multiplier."""
            filters *= width_coefficient
            new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_filters < 0.9 * filters:
                new_filters += divisor
            return int(new_filters)

        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(tf.math.ceil(tf.cast(depth_coefficient * repeats, tf.float32)))
        
        if input_tensor is None:
            img_input = tf.keras.layers.Input(shape = input_shape)
        else:
            if not tf.keras.backend.is_keras_tensor(input_tensor):
                img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
            else:
                img_input = input_tensor

        # Build stem
        x = img_input
        x = tf.keras.layers.ZeroPadding2D(padding=correct_pad(x, 3),
                                          name='stem_conv_pad')(x)
        #filters = round_filters(32) #efficientnet lite > fixed feature
        x = tf.keras.layers.Conv2D(32, 3,
                                   strides=2,
                                   padding='valid',
                                   use_bias=False,
                                   kernel_initializer=conv_kernel_initializer,
                                   name='stem_conv')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, name='stem_bn')(x)
        x = tf.keras.layers.Activation(activation_fn, name='stem_activation')(x)

        # Build blocks
        from copy import deepcopy
        blocks_args = deepcopy(blocks_args)

        b = 0
        blocks = float(sum(args['repeats'] for args in blocks_args))
        for (i, args) in enumerate(blocks_args):
            assert args['repeats'] > 0
            # Update block input and output filters based on depth multiplier.
            args['filters_in'] = round_filters(args['filters_in'])
            args['filters_out'] = round_filters(args['filters_out'])
            
            #repeats = round_repeats(args.pop('repeats')) #efficientnet lite > repeats condition add
            repeats = args.pop("repeats") if (i == 0 or i == (len(blocks_args) - 1)) else round_repeats(args.pop("repeats"))

            for j in range(repeats):
                # The first block needs to take care of stride and filter size increase.
                if j > 0:
                    args['strides'] = 1
                    args['filters_in'] = args['filters_out']
                x = block(x, activation_fn, drop_connect_rate * b / blocks,
                          name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
                b += 1
        
        # Build top
        #filters = round_filters(1280) #efficientnet lite > fixed feature
        x = tf.keras.layers.Conv2D(1280, 1,
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=conv_kernel_initializer,
                                   name='top_conv')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, name='top_bn')(x)
        x = tf.keras.layers.Activation(activation_fn, name='top_activation')(x)
        if include_top:
            x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            if dropout_rate > 0:
                x = tf.keras.layers.Dropout(dropout_rate, name='top_dropout')(x)
            x = tf.keras.layers.Dense(classes,
                             activation='softmax',
                             kernel_initializer=dense_kernel_initializer,
                             name='probs')(x)
        else:
            if pooling == 'avg':
                x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            elif pooling == 'max':
                x = tf.keras.layers.GlobalMaxPooling2D(name='max_pool')(x)
        
        model = tf.keras.Model(img_input, x)
        if weights is not None:
            model.load_weights(weights)
        return model
except:
    pass

effnet_lite_urls = {
    "effnet_lite_b0":"https://tfhub.dev/tensorflow/efficientnet/lite0/classification/2",
    "effnet_lite_b1":"https://tfhub.dev/tensorflow/efficientnet/lite1/classification/2",
    "effnet_lite_b2":"https://tfhub.dev/tensorflow/efficientnet/lite2/classification/2",
    "effnet_lite_b3":"https://tfhub.dev/tensorflow/efficientnet/lite3/classification/2",
    "effnet_lite_b4":"https://tfhub.dev/tensorflow/efficientnet/lite4/classification/2"
}

def load_weight(model, url):
    """
    https://tfhub.dev/tensorflow/efficientnet/lite0/classification/2
    https://tfhub.dev/tensorflow/efficientnet/lite1/classification/2
    https://tfhub.dev/tensorflow/efficientnet/lite2/classification/2
    https://tfhub.dev/tensorflow/efficientnet/lite3/classification/2
    https://tfhub.dev/tensorflow/efficientnet/lite4/classification/2
    """
    try:
        import tensorflow_hub as hub
        with tf.device("/cpu:0"):
            mod = hub.load(url)
    except:
        print("If you want to use 'EfficientNet-Lite Weight', please install 'tensorflow_hub'")
        return model
    for w, new_w in zip(model.weights, mod.variables):
        tf.keras.backend.set_value(w, new_w.numpy())
    return model

def effnet_lite_b0(x, activation = tf.nn.relu6, weights = "imagenet", indices = None):
    try:
        efficientnet_lite
    except:
        print("If you want to use 'EfficientNet-Lite', please install 'keras'")
        return
    hub_weight = False
    if weights == "imagenet":
        hub_weight = True
        weights = None
    model = efficientnet_lite(1.0, 1.0, 224, 0.2, activation_fn = activation, input_tensor = x, include_top = False, weights = weights)
    if hub_weight:
        model = load_weight(model, effnet_lite_urls["effnet_lite_b0"])
    layers = ["block2b_add", "block3b_add", "block5c_add", "block7a_project_bn"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def effnet_lite_b1(x, activation = tf.nn.relu6, weights = "imagenet", indices = None):
    try:
        efficientnet_lite
    except:
        print("If you want to use 'EfficientNet-Lite', please install 'keras'")
        return
    hub_weight = False
    if weights == "imagenet":
        hub_weight = True
        weights = None
    model = efficientnet_lite(1.0, 1.1, 240, 0.2, activation_fn = activation, input_tensor = x, include_top = False, weights = weights)
    if hub_weight:
        model = load_weight(model, effnet_lite_urls["effnet_lite_b1"])
    layers = ["block2c_add", "block3c_add", "block5d_add", "block7a_project_bn"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def effnet_lite_b2(x, activation = tf.nn.relu6, weights = "imagenet", indices = None):
    try:
        efficientnet_lite
    except:
        print("If you want to use 'EfficientNet-Lite', please install 'keras'")
        return
    hub_weight = False
    if weights == "imagenet":
        hub_weight = True
        weights = None
    model = efficientnet_lite(1.1, 1.2, 260, 0.3, activation_fn = activation, input_tensor = x, include_top = False, weights = weights)
    if hub_weight:
        model = load_weight(model, effnet_lite_urls["effnet_lite_b2"])
    layers = ["block2c_add", "block3c_add", "block5d_add", "block7a_project_bn"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def effnet_lite_b3(x, activation = tf.nn.relu6, weights = "imagenet", indices = None):
    try:
        efficientnet_lite
    except:
        print("If you want to use 'EfficientNet-Lite', please install 'keras'")
        return
    hub_weight = False
    if weights == "imagenet":
        hub_weight = True
        weights = None
    model = efficientnet_lite(1.2, 1.4, 280, 0.3, activation_fn = activation, input_tensor = x, include_top = False, weights = weights)
    if hub_weight:
        model = load_weight(model, effnet_lite_urls["effnet_lite_b3"])
    layers = ["block2c_add", "block3c_add", "block5e_add", "block7a_project_bn"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def effnet_lite_b4(x, activation = tf.nn.relu6, weights = "imagenet", indices = None):
    try:
        efficientnet_lite
    except:
        print("If you want to use 'EfficientNet-Lite', please install 'keras'")
        return
    hub_weight = False
    if weights == "imagenet":
        hub_weight = True
        weights = None
    model = efficientnet_lite(1.4, 1.8, 300, 0.3, activation_fn = activation, input_tensor = x, include_top = False, weights = weights)
    if hub_weight:
        model = load_weight(model, effnet_lite_urls["effnet_lite_b4"])
    layers = ["block2d_add", "block3d_add", "block5f_add", "block7a_project_bn"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def effnet_v2_b0(x, weights = "imagenet", indices = None):
    try:
        tf.keras.applications.EfficientNetV2B0
    except:
        print("If you want to use 'EfficientNetV2', please install 'tensorflow 2.8▲'")
        return
    model = tf.keras.applications.EfficientNetV2B0(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2b_add", "block3b_add", "block5e_add", "block6h_add"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature
    
def effnet_v2_b1(x, weights = "imagenet", indices = None):
    try:
        tf.keras.applications.EfficientNetV2B1
    except:
        print("If you want to use 'EfficientNetV2', please install 'tensorflow 2.8▲'")
        return
    model = tf.keras.applications.EfficientNetV2B1(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2c_add", "block3c_add", "block5f_add", "block6i_add"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature
    
def effnet_v2_b2(x, weights = "imagenet", indices = None):
    try:
        tf.keras.applications.EfficientNetV2B2
    except:
        print("If you want to use 'EfficientNetV2', please install 'tensorflow 2.8▲'")
        return
    model = tf.keras.applications.EfficientNetV2B2(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2c_add", "block3c_add", "block5f_add", "block6j_add"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature
    
def effnet_v2_b3(x, weights = "imagenet", indices = None):
    try:
        tf.keras.applications.EfficientNetV2B3
    except:
        print("If you want to use 'EfficientNetV2', please install 'tensorflow 2.8▲'")
        return
    model = tf.keras.applications.EfficientNetV2B3(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2c_add", "block3c_add", "block5g_add", "block6l_add"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature
    
def effnet_v2_s(x, weights = "imagenet", indices = None):
    try:
        tf.keras.applications.EfficientNetV2S
    except:
        print("If you want to use 'EfficientNetV2', please install 'tensorflow 2.8▲'")
        return
    model = tf.keras.applications.EfficientNetV2S(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2d_add", "block3d_add", "block5i_add", "block6o_add"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature
    
def effnet_v2_m(x, weights = "imagenet", indices = None):
    try:
        tf.keras.applications.EfficientNetV2M
    except:
        print("If you want to use 'EfficientNetV2', please install 'tensorflow 2.8▲'")
        return
    model = tf.keras.applications.EfficientNetV2M(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2e_add", "block3e_add", "block5n_add", "block7e_add"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature
    
def effnet_v2_l(x, weights = "imagenet", indices = None):
    try:
        tf.keras.applications.EfficientNetV2L
    except:
        print("If you want to use 'EfficientNetV2', please install 'tensorflow 2.8▲'")
        return
    model = tf.keras.applications.EfficientNetV2L(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2g_add", "block3g_add", "block5s_add", "block7g_add"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature