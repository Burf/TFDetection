import tensorflow as tf

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

class ClassNet(tf.keras.layers.Layer):
    def __init__(self, n_anchor, n_class = 21, n_feature = 224, n_depth = 4, concat = True, convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu, **kwargs):
        super(ClassNet, self).__init__(**kwargs)   
        self.n_anchor = n_anchor
        self.n_class = n_class
        self.n_feature = n_feature
        self.n_depth = n_depth
        self.concat = concat
        self.convolution = convolution
        self.normalize = normalize
        self.activation = activation

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        self.convs = [self.convolution(self.n_feature, 3, padding = "same", name = "depth{0}_conv".format(i + 1)) for i in range(self.n_depth)]
        if self.normalize is not None:
            self.norms = [[self.normalize(name = "depth{0}_norm{1}".format(i + 1, j + 1)) for j in range(len(input_shape))] for i in range(self.n_depth)]
        self.acts = [tf.keras.layers.Activation(self.activation, name = "depth{0}_act".format(i + 1)) for i in range(self.n_depth)]
        self.head = self.convolution(self.n_anchor * self.n_class, 3, padding = "same", name = "head")
        self.reshape = tf.keras.layers.Reshape([-1, self.n_class], name = "head_reshape")
        self.act = tf.keras.layers.Activation(tf.keras.activations.sigmoid, name = "logits")
        if self.concat and 1 < len(input_shape):
            self.post = tf.keras.layers.Concatenate(axis = -2, name = "logits_concat")

    def call(self, inputs, feature = False):
        if not isinstance(inputs, list):
            inputs = [inputs]
        out = []
        features = []
        for j, x in enumerate(inputs):
            for i in range(self.n_depth):
                x = self.convs[i](x)
                if self.normalize is not None:
                    x = self.norms[i][j](x)
                x = self.acts[i](x)
            features.append(x)
            x = self.act(self.reshape(self.head(x)))
            out.append(x)
        if len(out) == 1:
            out = out[0]
        elif self.concat:
            out = self.post(out)
        if feature:
            out = [out, features]
        return out
    
    def get_config(self):
        config = super(ClassNet, self).get_config()
        config["n_anchor"] = self.n_anchor
        config["n_class"] = self.n_class
        config["n_feature"] = self.n_feature
        config["n_depth"] = self.n_depth
        config["concat"] = self.concat
        config["convolution"] = self.convolution
        config["normalize"] = self.normalize
        config["activation"] = self.activation
        return config

class BoxNet(tf.keras.layers.Layer):
    def __init__(self, n_anchor, n_feature = 224, n_depth = 4, concat = True, convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu, **kwargs):
        super(BoxNet, self).__init__(**kwargs)   
        self.n_anchor = n_anchor
        self.n_feature = n_feature
        self.n_depth = n_depth
        self.concat = concat
        self.convolution = convolution
        self.normalize = normalize
        self.activation = activation

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        self.convs = [self.convolution(self.n_feature, 3, padding = "same", name = "depth{0}_conv".format(i + 1)) for i in range(self.n_depth)]
        if self.normalize is not None:
            self.norms = [[self.normalize(name = "depth{0}_norm{1}".format(i + 1, j + 1)) for j in range(len(input_shape))] for i in range(self.n_depth)]
        self.acts = [tf.keras.layers.Activation(self.activation, name = "depth{0}_act".format(i + 1)) for i in range(self.n_depth)]
        self.head = self.convolution(self.n_anchor * 4, 3, padding = "same", name = "head")
        self.reshape = tf.keras.layers.Reshape([-1, 4], name = "regress")

        if self.concat and 1 < len(input_shape):
            self.post = tf.keras.layers.Concatenate(axis = -2, name = "regress_concat")

    def call(self, inputs, feature = False):
        if not isinstance(inputs, list):
            inputs = [inputs]
        out = []
        features = []
        for j, x in enumerate(inputs):
            for i in range(self.n_depth):
                x = self.convs[i](x)
                if self.normalize is not None:
                    x = self.norms[i][j](x)
                x = self.acts[i](x)
            features.append(x)
            x = self.reshape(self.head(x))
            out.append(x)
        if len(out) == 1:
            out = out[0]
        elif self.concat:
            out = self.post(out)
        if feature:
            out = [out, features]
        return out
    
    def get_config(self):
        config = super(BoxNet, self).get_config()
        config["n_anchor"] = self.n_anchor
        config["n_feature"] = self.n_feature
        config["n_depth"] = self.n_depth
        config["concat"] = self.concat
        config["convolution"] = self.convolution
        config["normalize"] = self.normalize
        config["activation"] = self.activation
        return config