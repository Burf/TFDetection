import tensorflow as tf

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def head_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def normalize(axis = -1, momentum = 0.9, epsilon = 1e-5, **kwargs):
    return tf.keras.layers.BatchNormalization(axis = axis, momentum = momentum, epsilon = epsilon, **kwargs)

def fcn_head(feature, n_class = 35, n_feature = 512, n_depth = 2, dropout_rate = 0.1,
             logits_convolution = head_conv, logits_activation = tf.keras.activations.softmax,
             convolution = conv, normalize = normalize, activation = tf.keras.activations.relu,
             prefix = ""):
    #https://arxiv.org/pdf/1411.4038.pdf
    if isinstance(feature, (tuple, list)):
        feature = feature[-1]
    prefix = "{0}_".format(prefix) if len(prefix) != 0 else prefix
        
    out = feature
    for index in range(n_depth):
        out = convolution(n_feature, 3, padding = "same", use_bias = normalize is None, name = "{0}feature_conv{1}".format(prefix, index + 1))(out)
        if normalize is not None:
            out = normalize(name = "{0}feature_norm{1}".format(prefix, index + 1))(out)
        out = tf.keras.layers.Activation(activation, name = "{0}feature_act{1}".format(prefix, index + 1))(out)
    
    if 0 < n_depth:
        out = tf.keras.layers.Concatenate(axis = -1, name = "{0}post_concat".format(prefix))([out, feature])
        out = convolution(n_feature, 3, padding = "same", use_bias = normalize is None, name = "{0}post_conv".format(prefix))(out)
        if normalize is not None:
            out = normalize(name = "{0}post_norm".format(prefix))(out)
        out = tf.keras.layers.Activation(activation, name = "{0}post_act".format(prefix))(out)
    
    if 0 < dropout_rate:
        out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = logits_convolution(n_class, 1, use_bias = True, name = "{0}logits".format(prefix))(out)
    out = tf.keras.layers.Activation(logits_activation if logits_activation is not None else tf.keras.activations.linear, dtype = tf.float32, name = "{0}logits_act".format(prefix))(out)
    return out