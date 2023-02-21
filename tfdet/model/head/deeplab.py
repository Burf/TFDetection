import tensorflow as tf

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def head_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def normalize(axis = -1, momentum = 0.9, epsilon = 1e-5, **kwargs):
    return tf.keras.layers.BatchNormalization(axis = axis, momentum = momentum, epsilon = epsilon, **kwargs)

class AtrousSpatialPyramidPooling(tf.keras.layers.Layer):
    def __init__(self, rate = [12, 24, 36], n_feature = 512, method = "bilinear", convolution = conv, normalize = normalize, activation = tf.keras.activations.relu, **kwargs):
        super(AtrousSpatialPyramidPooling, self).__init__(**kwargs)
        self.rate = rate
        self.n_feature = n_feature
        self.method = method
        self.convolution = convolution
        self.normalize = normalize
        self.activation = activation
        
    def build(self, input_shape):
        self.extract_feature = []
        
        for dilation_rate in self.rate:
            kernel_size = 3 if dilation_rate != 1 else 1
            layer = []
            layer.append(self.convolution(self.n_feature, kernel_size, dilation_rate = dilation_rate, padding = "same", use_bias = self.normalize is None))
            if self.normalize is not None:
                layer.append(self.normalize())
            layer.append(tf.keras.layers.Activation(self.activation))
            self.extract_feature.append(layer)
        
        layer = []
        layer.append(tf.keras.layers.GlobalAveragePooling2D())
        layer.append(tf.keras.layers.Reshape([1, 1, input_shape[-1]]))
        layer.append(self.convolution(self.n_feature, 1, use_bias = self.normalize is None))
        if self.normalize is not None:
            layer.append(self.normalize())
        layer.append(tf.keras.layers.Activation(self.activation))
        layer.append(tf.keras.layers.Lambda(lambda args, target_size, method: tf.image.resize(args, target_size, method = self.method), arguments = {"target_size":input_shape[1:3], "method":self.method}))
        self.extract_feature.append(layer)
        self.concat = tf.keras.layers.Concatenate(axis = -1)

        layer = []
        layer.append(self.convolution(self.n_feature, 1, use_bias = self.normalize is None))
        if self.normalize is not None:
            layer.append(self.normalize())
        layer.append(tf.keras.layers.Activation(self.activation))
        self.conv_block = layer

    def call(self, inputs):
        out = []
        for module in self.extract_feature:
            o = inputs
            for layer in module:
                o = layer(o)
            out.append(o)
        
        out = self.concat(out)
        for layer in self.conv_block:
            out = layer(out)
        return out
    
    def get_config(self):
        config = super(AtrousSpatialPyramidPooling, self).get_config()
        config["rate"] = self.rate
        config["n_feature"] = self.n_feature
        config["method"] = self.method
        return config

def deeplab_v3_head(x, n_class = 35, rate = [1, 12, 24, 36], n_feature = 512, n_low_feature = 48, method = "bilinear", dropout_rate = 0.1,
                    logits_convolution = head_conv, logits_activation = tf.keras.activations.softmax,
                    convolution = conv, normalize = normalize, activation = tf.keras.activations.relu):
    if not isinstance(x, list):
        x = [x]
    low_features, feature = x[:-1], x[-1]
    
    #aspp
    feature = AtrousSpatialPyramidPooling(rate, n_feature, method, convolution = convolution, normalize = normalize, activation = activation, name = "aspp_feature" if len(low_features) == 0 else "aspp_feature1")(feature)

    #decoder - deeplab v3 plus
    upsample = tf.keras.layers.Lambda(lambda args: tf.image.resize(args[0], args[1], method = method), name = "feature_upsample")
    for index, low_feature in enumerate(low_features[::-1]):
        low_feature = convolution(n_low_feature * (2 ** (len(low_features) - (index + 1))), 1, padding = "same", use_bias = normalize is None, name = "low_feature{0}_conv".format(len(low_features) - index))(low_feature)
        if normalize is not None:
            low_feature = normalize(name = "low_feature{0}_norm".format(len(low_features) - index))(low_feature)
        low_feature = tf.keras.layers.Activation(activation, name = "low_feature{0}_act".format(len(low_features) - index))(low_feature)
        feature = upsample([feature, tf.shape(low_feature)[1:3]])
        feature = tf.keras.layers.Concatenate(axis = -1, name = "feature{0}".format(index + 2))([low_feature, feature])
        feature = convolution(n_feature, 3, padding = "same", use_bias = normalize is None, name = "feature{0}_conv".format(index + 2))(feature)
        if normalize is not None:
            feature = normalize(name = "feature{0}_norm".format(index + 2))(feature)
        feature = tf.keras.layers.Activation(activation, name = "feature{0}_act".format(index + 2))(feature)
    
    #head
    out = convolution(n_feature, 3, padding = "same", use_bias = normalize is None, name = "head_conv")(feature)
    if normalize is not None:
        out = normalize(name = "head_norm")(out)
    out = tf.keras.layers.Activation(activation, name = "head_act")(out)
    
    if 0 < dropout_rate:
        out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = logits_convolution(n_class, 1, use_bias = True, name = "logits")(out)
    out = tf.keras.layers.Activation(logits_activation if logits_activation is not None else tf.keras.activations.linear, dtype = tf.float32, name = "logits_act")(out)
    return out