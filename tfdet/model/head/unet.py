import tensorflow as tf

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def unet_head(x, n_class = 35, n_feature = 64, n_depth = 5, method = "bilinear", logits_activation = tf.keras.activations.sigmoid, convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    if isinstance(x, list):
        out, feature = x[-1], x[:-1]
        n_feature *= (2 ** (len(x) - 1))
        n_depth = len(x)
    else:
        out = x
        feature = []
        for index in range(n_depth):
            _n_feature = n_feature * (2 ** index)
            out = convolution(_n_feature, 3, padding = "same", use_bias = normalize is None, name = "downsample{0}_conv1".format(index + 1))(out)
            if normalize is not None:
                out = normalize(name = "downsample{0}_norm1".format(index + 1))(out)
            out = tf.keras.layers.Activation(activation, name = "downsample{0}_act1".format(index + 1))(out)
            out = convolution(_n_feature, 3, padding = "same", use_bias = normalize is None, name = "downsample{0}_conv2".format(index + 1))(out)
            if normalize is not None:
                out = normalize(name = "downsample{0}_norm2".format(index + 1))(out)
            out = tf.keras.layers.Activation(activation, name = "downsample{0}_act2".format(index + 1))(out)
            if index + 1 < n_depth:
                feature.append(out)
                out = tf.keras.layers.MaxPool2D(name = "downsample{0}_pooling".format(index + 1))(out)
            else:
                n_feature = _n_feature
            
    for index in range(n_depth - 1):
        _n_feature = n_feature / (2 ** (index + 1))
        prev_feature = feature[-(index + 1)]
        target_size = tf.keras.backend.int_shape(prev_feature)[1:3]
        out = tf.keras.layers.Lambda(lambda args, target_size, method: tf.image.resize(args, target_size, method = method), arguments = {"target_size":target_size, "method":method}, name = "upsample{0}_sampling".format(index + 1))(out)
        out = convolution(_n_feature, 3, padding = "same", use_bias = normalize is None, name = "upsample{0}_pre_conv".format(index + 1))(out)
        if normalize is not None:
            out = normalize(name = "upsample{0}_pre_norm".format(index + 1))(out)
        out = tf.keras.layers.Activation(activation, name = "upsample{0}_pre_act".format(index + 1))(out)
        out = tf.keras.layers.Concatenate(axis = -1, name = "upsample{0}_concat".format(index + 1))([prev_feature, out])
        
        out = convolution(_n_feature, 3, padding = "same", use_bias = normalize is None, name = "upsample{0}_conv1".format(index + 1))(out)
        if normalize is not None:
            out = normalize(name = "upsample{0}_norm1".format(index + 1))(out)
        out = tf.keras.layers.Activation(activation, name = "upsample{0}_act1".format(index + 1))(out)
        out = convolution(_n_feature, 3, padding = "same", use_bias = normalize is None, name = "upsample{0}_conv2".format(index + 1))(out)
        if normalize is not None:
            out = normalize(name = "upsample{0}_norm2".format(index + 1))(out)
        out = tf.keras.layers.Activation(activation, name = "upsample{0}_act2".format(index + 1))(out)
    out = convolution(n_class, 1, padding = "same", use_bias = True, activation = logits_activation, name = "logits")(out)
    return out

def unet_2plus_head(feature, n_class = 35, n_feature = 256, deep_supervision = False, method = "bilinear", logits_activation = tf.keras.activations.sigmoid, convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    if not isinstance(feature, list):
        feature = [feature]
    feature = list(feature)
    
    out = [feature.pop()]
    for i, x in zip(range(len(feature) - 1, -1, -1), reversed(feature)):
        target_size = tf.keras.backend.int_shape(x)[-3:-1]
        _out = [x]
        for j, o in enumerate(out):
            o = tf.keras.layers.Lambda(lambda args, target_size, method: tf.image.resize(args, target_size, method = method), arguments = {"target_size":target_size, "method":method}, name = "{0}_{1}_upsample".format(i + 1, j))(o)
            o = tf.keras.layers.Concatenate(axis = -1, name = "{0}_{1}_concat".format(i, j + 1))([o, x])
            o = convolution(n_feature, 3, padding = "same", use_bias = normalize is None, name = "{0}_{1}_post_conv".format(i, j + 1))(o)
            if normalize is not None:
                o = normalize(name = "{0}_{1}_post_norm".format(i, j + 1))(o)
            x = tf.keras.layers.Activation(activation, name = "{0}_{1}_post_act".format(i, j + 1))(o)
            _out.append(x)
        out = _out
        n_feature //= 2
    if deep_supervision:
        out = out[1:]
    else:
        out = out[-1:]
    out = [convolution(n_class, 3, padding = "same", activation = logits_activation, name = "logits{0}".format(i) if 1 < len(out) else "logits")(o) for i, o in enumerate(out)]
    if len(out) == 1:
        out = out[0]
    return out