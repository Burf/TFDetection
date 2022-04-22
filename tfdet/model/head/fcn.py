import tensorflow as tf

class UpsamplingFeature(tf.keras.layers.Layer):
    def __init__(self, concat = True, method = "bilinear", **kwargs):
        super(UpsamplingFeature, self).__init__(**kwargs)
        self.concat = concat
        self.method = method

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        self.resize = tf.keras.layers.Lambda(lambda args, target_size, method: tf.image.resize(args, target_size, method = method), arguments = {"target_size":input_shape[0][1:3], "method":self.method})
        self.post = tf.keras.layers.Concatenate(axis = -1)
        
    def call(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        out = [inputs[0]] + [self.resize(x) for x in inputs[1:]]
        
        if self.concat and 1 < len(out):
            out = self.post(out)
        elif len(out) == 1:
            out = out[0]
        return out
    
    def get_config(self):
        config = super(UpsamplingFeature, self).get_config()
        config["concat"] = self.concat
        config["method"] = self.method
        return config

def fcn(feature, n_class = 35, n_feature = 512, n_depth = 2, method = "bilinear", logits_activation = tf.keras.activations.sigmoid, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    #https://arxiv.org/pdf/1411.4038.pdf
    if not isinstance(feature, list):
        feature = [feature]
    
    out = feature = UpsamplingFeature(concat = True, method = method, name = "upsampling_feature")(feature)
    for index in range(n_depth):
        out = tf.keras.layers.Conv2D(n_feature, 3, padding = "same", use_bias = normalize is None, kernel_initializer = "he_normal", name = "feature_conv{0}".format(index + 1))(out)
        if normalize is not None:
            out = normalize(name = "feature_norm{0}".format(index + 1))(out)
        out = tf.keras.layers.Activation(activation, name = "feature_act{0}".format(index + 1))(out)
    
    if 0 < n_depth:
        out = tf.keras.layers.Concatenate(axis = -1, name = "post_concat")([out, feature])
        out = tf.keras.layers.Conv2D(n_feature, 3, padding = "same", use_bias = normalize is None, kernel_initializer = "he_normal", name = "post_conv")(out)
        if normalize is not None:
            out = normalize(name = "post_norm")(out)
        out = tf.keras.layers.Activation(activation, name = "post_act")(out)
    
    out = tf.keras.layers.Conv2D(n_class, 1, use_bias = True, activation = logits_activation, kernel_initializer = "he_normal", name = "logits")(out)
    return out