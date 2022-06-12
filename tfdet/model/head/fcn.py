import tensorflow as tf

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def fcn_head(feature, n_class = 35, n_feature = 512, n_depth = 2, logits_activation = tf.keras.activations.sigmoid, convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    #https://arxiv.org/pdf/1411.4038.pdf
    if isinstance(feature, list):
        feature = feature[-1]
    out = feature
    for index in range(n_depth):
        out = convolution(n_feature, 3, padding = "same", use_bias = normalize is None, name = "feature_conv{0}".format(index + 1))(out)
        if normalize is not None:
            out = normalize(name = "feature_norm{0}".format(index + 1))(out)
        out = tf.keras.layers.Activation(activation, name = "feature_act{0}".format(index + 1))(out)
    
    if 0 < n_depth:
        out = tf.keras.layers.Concatenate(axis = -1, name = "post_concat")([out, feature])
        out = convolution(n_feature, 3, padding = "same", use_bias = normalize is None, name = "post_conv")(out)
        if normalize is not None:
            out = normalize(name = "post_norm")(out)
        out = tf.keras.layers.Activation(activation, name = "post_act")(out)
    
    out = convolution(n_class, 1, use_bias = True, activation = logits_activation, name = "logits")(out)
    return out