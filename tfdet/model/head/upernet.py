import tensorflow as tf

from .pspnet import PoolingPyramidModule

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def head_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def normalize(axis = -1, momentum = 0.9, epsilon = 1e-5, **kwargs):
    return tf.keras.layers.BatchNormalization(axis = axis, momentum = momentum, epsilon = epsilon, **kwargs)

def upernet_head(feature, n_class = 35, n_feature = 512, pool_scale = [1, 2, 3, 6], max_pooling = False, method = "bilinear", dropout_rate = 0.1,
                 logits_convolution = head_conv, logits_activation = tf.keras.activations.softmax,
                 convolution = conv, normalize = normalize, activation = tf.keras.activations.relu):
    #https://arxiv.org/abs/1807.10221
    feature = [feature] if not isinstance(feature, (tuple, list)) else list(feature)
    
    out = []
    for i, x in enumerate(feature[:-1]):
        o = convolution(n_feature, 1, use_bias = normalize is None, name = "feature{0}_resample_conv".format(i + 1) if 1 < len(feature) else "feature_resample_conv")(x)
        if normalize is not None:
            o = normalize(name = "feature{0}_resample_norm".format(i + 1) if 1 < len(feature) else "feature_resample_norm")(o)
        o = tf.keras.layers.Activation(activation, name = "feature{0}_resample_act".format(i + 1) if 1 < len(feature) else "feature_resample_act")(o)
        out.append(o)
    
    psp_feature = PoolingPyramidModule(pool_scale, n_feature, max_pooling = max_pooling, method = method, convolution = convolution, normalize = normalize, activation = activation, name = "psp_feature_extract")(feature[-1])
    psp_feature = tf.keras.layers.Concatenate(axis = -1, name = "psp_feature")([feature[-1]] + psp_feature)
    psp_feature = convolution(n_feature, 3, padding = "same", use_bias = normalize is None, name = "psp_feature_conv")(psp_feature)
    if normalize is not None:
        psp_feature = normalize(name = "psp_feature_norm")(psp_feature)
    psp_feature = tf.keras.layers.Activation(activation, name = "psp_feature_act")(psp_feature)
    out.append(psp_feature)
    
    upsample = tf.keras.layers.Lambda(lambda args: tf.image.resize(args[0], args[1], method = method), name = "feature_upsample")
    for index in range(len(out) - 1, 0, -1):
        prev_size = tf.shape(out[index - 1])[1:3]
        out[index - 1] = tf.keras.layers.Add(name = "feature{0}_add".format(index))([out[index - 1], upsample([out[index], prev_size])])
    
    for index in range(len(out) - 1): #without psp feature
        o = convolution(n_feature, 3, padding = "same", use_bias = normalize is None, name = "feature{0}_post_conv".format(index + 1))(out[index])
        if normalize is not None:
            o = normalize(name = "feature{0}_post_norm".format(index + 1))(o)
        o = tf.keras.layers.Activation(activation, name = "feature{0}_post_act".format(index + 1))(o)
        out[index] = o
    
    target_size = tf.shape(out[0])[1:3]
    upsample = tf.keras.layers.Lambda(lambda args: tf.image.resize(args[0], args[1], method = method), name = "feature_post_upsample")
    out = [upsample([o, target_size]) if i != 0 else o for i, o in enumerate(out)]
    out = tf.keras.layers.Concatenate(axis = -1, name = "feature_post_concat")(out)
    
    out = convolution(n_feature, 3, padding = "same", use_bias = normalize is None, name = "head_conv")(out)
    if normalize is not None:
        out = normalize(name = "head_norm")(out)
    out = tf.keras.layers.Activation(activation, name = "head_act")(out)
                   
    if 0 < dropout_rate:
        out = tf.keras.layers.Dropout(dropout_rate)(out)                                                 
    out = logits_convolution(n_class, 1, use_bias = True, name = "logits")(out)
    out = tf.keras.layers.Activation(logits_activation if logits_activation is not None else tf.keras.activations.linear, dtype = tf.float32, name = "logits_act")(out)
    return out