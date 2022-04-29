import tensorflow as tf

from ..head.pspnet import PoolingPyramidModule

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def upernet(feature, n_class = 35, n_feature = 512, pool_scale = [1, 2, 3, 6], method = "bilinear", logits_activation = tf.keras.activations.sigmoid, convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    #https://arxiv.org/abs/1807.10221
    if not isinstance(feature, list):
        feature = [feature]
    
    out = []
    for i, x in enumerate(feature[:-1]):
        o = tf.keras.layers.Conv2D(n_feature, 1, use_bias = normalize is None, kernel_initializer = "he_normal", name = "feature{0}_resample_conv".format(i + 1) if 1 < len(feature) else "feature_resample_conv")(x)
        if normalize is not None:
            o = normalize(name = "feature{0}_resample_norm".format(i + 1) if 1 < len(feature) else "feature_resample_norm")(o)
        o = tf.keras.layers.Activation(activation, name = "feature{0}_resample_act".format(i + 1) if 1 < len(feature) else "feature_resample_act")(o)
        out.append(o)
    
    psp_feature = PoolingPyramidModule(pool_scale, n_feature, method = method, convolution = convolution, normalize = normalize, activation = activation, name = "psp_feature_extract")(feature[-1])
    psp_feature = tf.keras.layers.Concatenate(axis = -1, name = "psp_feature")([feature[-1]] + psp_feature)
    psp_feature = convolution(n_feature, 3, padding = "same", use_bias = normalize is None, name = "psp_feature_conv")(psp_feature)
    if normalize is not None:
        psp_feature = normalize(name = "psp_feature_norm")(psp_feature)
    psp_feature = tf.keras.layers.Activation(activation, name = "psp_feature_act")(psp_feature)
    out.append(psp_feature)
    
    for index in range(len(out) - 1, 0, -1):
        prev_size = tf.keras.backend.int_shape(out[index - 1])[1:3]
        upsample = tf.keras.layers.Lambda(lambda args, target_size, method: tf.image.resize(args, target_size, method = method), arguments = {"target_size":prev_size, "method":method}, name = "feature{0}_upsample".format(index + 1))(out[index])
        out[index - 1] = tf.keras.layers.Add(name = "feature{0}_add".format(index))([out[index - 1], upsample])
    
    for index in range(len(out) - 1): #without psp feature
        o = convolution(n_feature, 3, padding = "same", use_bias = normalize is None, name = "feature{0}_post_conv".format(index + 1))(out[index])
        if normalize is not None:
            o = normalize(name = "feature{0}_post_norm".format(index + 1))(o)
        o = tf.keras.layers.Activation(activation, name = "feature{0}_post_act".format(index + 1))(o)
        out[index] = o
    
    target_size = tf.keras.backend.int_shape(out[0])[1:3]
    resize = tf.keras.layers.Lambda(lambda args, target_size, method: tf.image.resize(args, target_size, method = method), arguments = {"target_size":target_size, "method":method}, name = "feature_post_upsample")
    out = [resize(o) if i != 0 else o for i, o in enumerate(out)]
    out = tf.keras.layers.Concatenate(axis = -1, name = "feature_post_concat")(out)
    
    out = convolution(n_feature, 3, padding = "same", use_bias = normalize is None, name = "head_conv")(out)
    if normalize is not None:
        out = normalize(name = "head_norm")(out)
    out = tf.keras.layers.Activation(activation, name = "head_act")(out)
    out = convolution(n_class, 1, use_bias = True, activation = logits_activation, name = "logits")(out)
    return out