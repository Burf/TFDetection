import tensorflow as tf
import numpy as np

from tfdet.core.anchor import generate_anchors
from tfdet.core.ops.initializer import PriorProbability

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def cls_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), bias_initializer = PriorProbability(probability = 0.01), **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def bbox_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

class ClassNet(tf.keras.layers.Layer):
    def __init__(self, n_anchor, n_class = 21, n_feature = 224, n_depth = 4, use_bias = None, concat = False,
                 logits_convolution = cls_conv, logits_activation = tf.keras.activations.sigmoid,
                 convolution = conv, normalize = None, activation = tf.keras.activations.relu, **kwargs):
        super(ClassNet, self).__init__(**kwargs)   
        self.n_anchor = n_anchor
        self.n_class = n_class
        self.n_feature = n_feature
        self.n_depth = n_depth
        self.use_bias = (normalize is None) if use_bias is None else use_bias
        self.concat = concat
        self.logits_convolution = logits_convolution
        self.logits_activation = logits_activation
        self.convolution = convolution
        self.normalize = normalize
        self.activation = activation

    def build(self, input_shape):
        if not isinstance(input_shape, (tuple, list)):
            input_shape = [input_shape]
        self.convs = [self.convolution(self.n_feature, 3, padding = "same", use_bias = self.use_bias, name = "depth{0}_conv".format(i + 1)) for i in range(self.n_depth)]
        if self.normalize is not None:
            self.norms = [[self.normalize(name = "depth{0}_norm{1}".format(i + 1, j + 1)) for j in range(len(input_shape))] for i in range(self.n_depth)]
        self.acts = [tf.keras.layers.Activation(self.activation, name = "depth{0}_act".format(i + 1)) for i in range(self.n_depth)]
        self.head = self.logits_convolution(self.n_anchor * self.n_class, 3, padding = "same", name = "head")
        self.reshape = tf.keras.layers.Reshape([-1, self.n_class], name = "head_reshape")
        if self.concat and 1 < len(input_shape):
            self.post = tf.keras.layers.Concatenate(axis = -2, name = "logits_concat")
        self.act = tf.keras.layers.Activation(self.logits_activation, dtype = tf.float32, name = "logits")

    def call(self, inputs, feature = False):
        if not isinstance(inputs, (tuple, list)):
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
        if isinstance(out, (tuple, list)):
            out = [self.act(o) for o in out]
        else:
            out = self.act(out)
        if feature:
            out = [out, features]
        return out
    
    def get_config(self):
        config = super(ClassNet, self).get_config()
        config["n_anchor"] = self.n_anchor
        config["n_class"] = self.n_class
        config["n_feature"] = self.n_feature
        config["n_depth"] = self.n_depth
        config["use_bias"] = self.use_bias
        config["concat"] = self.concat
        return config

class BoxNet(tf.keras.layers.Layer):
    def __init__(self, n_anchor, n_feature = 224, n_depth = 4, use_bias = None, concat = False, 
                 logits_convolution = bbox_conv, logits_activation = tf.keras.activations.linear,
                 convolution = conv, normalize = None, activation = tf.keras.activations.relu, **kwargs):
        super(BoxNet, self).__init__(**kwargs)   
        self.n_anchor = n_anchor
        self.n_feature = n_feature
        self.n_depth = n_depth
        self.use_bias = (normalize is None) if use_bias is None else use_bias
        self.concat = concat
        self.logits_convolution = logits_convolution
        self.logits_activation = logits_activation
        self.convolution = convolution
        self.normalize = normalize
        self.activation = activation

    def build(self, input_shape):
        if not isinstance(input_shape, (tuple, list)):
            input_shape = [input_shape]
        self.convs = [self.convolution(self.n_feature, 3, padding = "same", use_bias = self.use_bias, name = "depth{0}_conv".format(i + 1)) for i in range(self.n_depth)]
        if self.normalize is not None:
            self.norms = [[self.normalize(name = "depth{0}_norm{1}".format(i + 1, j + 1)) for j in range(len(input_shape))] for i in range(self.n_depth)]
        self.acts = [tf.keras.layers.Activation(self.activation, name = "depth{0}_act".format(i + 1)) for i in range(self.n_depth)]
        self.head = self.logits_convolution(self.n_anchor * 4, 3, padding = "same", name = "head")
        self.reshape = tf.keras.layers.Reshape([-1, 4], name = "regress")
        if self.concat and 1 < len(input_shape):
            self.post = tf.keras.layers.Concatenate(axis = -2, name = "regress_concat")
        self.act = tf.keras.layers.Activation(self.logits_activation, dtype = tf.float32, name = "regress_act")

    def call(self, inputs, feature = False):
        if not isinstance(inputs, (tuple, list)):
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
        if isinstance(out, (tuple, list)):
            out = [self.act(o) for o in out]
        else:
            out = self.act(out)
        if feature:
            out = [out, features]
        return out
    
    def get_config(self):
        config = super(BoxNet, self).get_config()
        config["n_anchor"] = self.n_anchor
        config["n_feature"] = self.n_feature
        config["n_depth"] = self.n_depth
        config["use_bias"] = self.use_bias
        config["concat"] = self.concat
        return config

def retina_head(feature, n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4, use_bias = None,
                scale = [32, 64, 128, 256, 512], ratio = [0.5, 1, 2], octave = 3,
                cls_convolution = cls_conv, cls_activation = tf.keras.activations.sigmoid,
                bbox_convolution = bbox_conv, bbox_activation = tf.keras.activations.linear,
                convolution = conv, normalize = None, activation = tf.keras.activations.relu):
    if tf.is_tensor(image_shape) and 2 < tf.keras.backend.ndim(image_shape) or (not tf.is_tensor(image_shape) and 2 < np.ndim(image_shape)):
        image_shape = tf.shape(image_shape) if tf.keras.backend.int_shape(image_shape)[-3] is None else tf.keras.backend.int_shape(image_shape)
    if 2 < np.shape(image_shape)[0]:
        image_shape = image_shape[-3:-1]
    if not isinstance(feature, (tuple, list)):
        feature = [feature]
    if np.ndim(scale) == 0:
        scale = [[scale]]
    elif np.ndim(scale) == 1:
        scale = np.expand_dims(scale, axis = -1)
    if np.ndim(ratio) == 0:
        ratio = [ratio]
    feature = list(feature)
    
    if np.ndim(scale) == 2 and np.shape(scale)[-1] == 1:
        scale = np.multiply(scale, [[2 ** (o / octave) for o in range(octave)]])
    n_anchor = len(scale) * len(ratio)
    if (len(feature) % len(scale)) == 0:
        n_anchor = len(scale[0]) * len(ratio)
    y_pred = ClassNet(n_anchor, n_class, n_feature, n_depth, use_bias,
                          logits_convolution = cls_convolution, logits_activation = cls_activation,
                          convolution = convolution, normalize = normalize, activation = activation,
                          concat = False, name = "class_net")(feature)
    bbox_pred = BoxNet(n_anchor, n_feature, n_depth, use_bias,
                       logits_convolution = bbox_convolution, logits_activation = bbox_activation,
                       convolution = convolution, normalize = normalize, activation = activation,
                       concat = False, name = "box_net")(feature)
    anchors = generate_anchors(feature, image_shape, scale, ratio, normalize = True, auto_scale = True, concat = False, dtype = tf.float32)
    return y_pred, bbox_pred, anchors