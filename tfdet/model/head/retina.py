import tensorflow as tf
import numpy as np

from tfdet.core.anchor import generate_anchors

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
        return config

def retina_head(feature, n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4,
                scale = [0.03125, 0.0625, 0.125, 0.25, 0.5], ratio = [0.5, 1, 2], auto_scale = True,
                convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    if tf.is_tensor(image_shape) and 2 < tf.keras.backend.ndim(image_shape) or (not tf.is_tensor(image_shape) and 2 < np.ndim(image_shape)):
        image_shape = tf.shape(image_shape) if tf.keras.backend.int_shape(image_shape)[-3] is None else tf.keras.backend.int_shape(image_shape)
    if 2 < np.shape(image_shape)[0]:
        image_shape = image_shape[-3:-1]
    if not isinstance(feature, list):
        feature = [feature]
    if np.ndim(scale) == 0:
        scale = [scale]
    if np.ndim(ratio) == 0:
        ratio = [ratio]
    feature = list(feature)
    
    n_anchor = len(scale) * len(ratio)
    if np.ndim(scale) == 2:
        n_anchor = len(scale[0]) * len(ratio)
    elif auto_scale and (len(scale) % len(feature)) == 0:
        n_anchor = (len(scale) // len(feature)) * len(ratio)
    logits = ClassNet(n_anchor, n_class, n_feature, n_depth, convolution = convolution, normalize = normalize, activation = activation, name = "class_net")(feature)
    regress = BoxNet(n_anchor, n_feature, n_depth, convolution = convolution, normalize = normalize, activation = activation, name = "box_net")(feature)
    anchors = generate_anchors(feature, image_shape, scale, ratio, normalize = True, auto_scale = auto_scale, dtype = logits.dtype)

    #valid_flags = tf.logical_and(tf.less_equal(anchors[..., 2], 1),
    #                             tf.logical_and(tf.less_equal(anchors[..., 3], 1),
    #                                            tf.logical_and(tf.greater_equal(anchors[..., 0], 0),
    #                                                           tf.greater_equal(anchors[..., 1], 0))))
    ##valid_indices = tf.range(tf.shape(anchors)[0])[valid_flags]
    #valid_indices = tf.where(valid_flags)[:, 0]
    #logits = tf.gather(logits, valid_indices, axis = 1)
    #regress = tf.gather(regress, valid_indices, axis = 1)
    #anchors = tf.gather(anchors, valid_indices)
    return logits, regress, anchors