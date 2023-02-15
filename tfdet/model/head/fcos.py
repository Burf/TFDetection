import tensorflow as tf
import numpy as np

from tfdet.core.anchor import generate_points
from tfdet.core.ops.initializer import PriorProbability
from .retina import ClassNet, BoxNet

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def cls_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), bias_initializer = PriorProbability(probability = 0.01), **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def bbox_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def conf_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def group_normalize(groups = 32, epsilon = 1e-5, momentum = 0.9, **kwargs):
    layer = None
    try:
        layer = tf.keras.layers.GroupNormalization(groups = groups, epsilon = epsilon, **kwargs)
    except:
        try:
            import tensorflow_addons as tfa
            layer = tfa.layers.GroupNormalization(groups = groups, epsilon = epsilon, **kwargs)
        except:
            pass
    if layer is None:
        print("If you want to use 'GroupNormalization', please install 'tensorflow 2.11▲ or tensorflow_addons'")
        layer = tf.keras.layers.BatchNormalization(momentum = momentum, epsilon = epsilon, **kwargs)
    return layer

class CenternessNet(tf.keras.layers.Layer):
    def __init__(self, n_anchor, use_bias = None, concat = False, convolution = conf_conv, normalize = None, activation = tf.keras.activations.sigmoid, **kwargs):
        super(CenternessNet, self).__init__(**kwargs)
        self.n_anchor = n_anchor
        self.use_bias = (normalize is None) if use_bias is None else use_bias
        self.concat = concat
        self.activation = activation
        self.convolution = convolution
        self.normalize = normalize

    def build(self, input_shape):
        if not isinstance(input_shape, (tuple, list)):
            input_shape = [input_shape]
        
        self.layers = [self.convolution(self.n_anchor, 3, padding = "same", use_bias = self.use_bias, name = "head")]
        if self.normalize is not None:
            self.layers.append(self.normalize(name = "norm"))
        self.layers.append(tf.keras.layers.Reshape([-1, 1], name = "reshape"))
        if self.concat and 1 < len(input_shape):
            self.post = tf.keras.layers.Concatenate(axis = -2, name = "logits_concat")
        self.act = tf.keras.layers.Activation(self.activation, dtype = tf.float32, name = "logits")

    def call(self, inputs):
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        out = []
        for x in inputs:
            for l in self.layers:
                x = l(x)
            out.append(x)
        if len(out) == 1:
            out = out[0]
        elif self.concat:
            out = self.post(out)
        if isinstance(out, (tuple, list)):
            out = [self.act(o) for o in out]
        else:
            out = self.act(out)
        return out
    
    def get_config(self):
        config = super(BoxNet, self).get_config()
        config["n_anchor"] = self.n_anchor
        config["use_bias"] = self.use_bias
        config["concat"] = self.concat
        return config
    
class Scale(tf.keras.layers.Layer):
    def __init__(self, value = 1., **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.value = value

    def build(self, input_shape):
        if not isinstance(input_shape, (tuple, list)):
            input_shape = [input_shape]
        self.w = [self.add_weight(name = "weight{0}".format(index + 1) if 1 < len(input_shape) else "weight",
                                 shape = (1,),
                                 initializer = tf.keras.initializers.constant(self.value),
                                 trainable = self.trainable,
                                 dtype = self.dtype) for index in range(len(input_shape))]

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        out = [inputs[index] * self.w[index] for index in range(len(inputs))]
        if len(out) == 1:
            out = out[0]
        return out

    def get_config(self):
        config = super(Scale, self).get_config()
        config["value"] = self.value
        return config
        
def fcos_head(feature, n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4, use_bias = None,
              centerness = True,
              cls_convolution = cls_conv, cls_activation = tf.keras.activations.sigmoid,
              bbox_convolution = bbox_conv, bbox_activation = tf.keras.activations.linear,
              conf_convolution = conf_conv, conf_normalize = None, conf_activation = tf.keras.activations.sigmoid,
              convolution = conv, normalize = group_normalize, activation = tf.keras.activations.relu):
    if tf.is_tensor(image_shape) and 2 < tf.keras.backend.ndim(image_shape) or (not tf.is_tensor(image_shape) and 2 < np.ndim(image_shape)):
        image_shape = tf.shape(image_shape) if tf.keras.backend.int_shape(image_shape)[-3] is None else tf.keras.backend.int_shape(image_shape)
    if 2 < np.shape(image_shape)[0]:
        image_shape = image_shape[-3:-1]
    if not isinstance(feature, (tuple, list)):
        feature = [feature]
    feature = list(feature)
    
    n_anchor = 1
    y_pred, logits_feature = ClassNet(n_anchor, n_class, n_feature, n_depth, use_bias,
                                      logits_convolution = cls_convolution, logits_activation = cls_activation,
                                      convolution = convolution, normalize = normalize, activation = activation,
                                      concat = False, name = "class_net")(feature, feature = True)
    bbox_pred = BoxNet(n_anchor, n_feature, n_depth, use_bias,
                       logits_convolution = bbox_convolution, logits_activation = bbox_activation,
                       convolution = convolution, normalize = normalize, activation = activation,
                       concat = False, name = "box_net")(feature)
    bbox_pred = Scale(1., dtype = tf.float32, name = "box_net_with_scale_factor")(bbox_pred)
    if not isinstance(bbox_pred, (tuple, list)):
        bbox_pred = [bbox_pred]
    act = tf.keras.layers.Activation(tf.exp, dtype = tf.float32, name = "box_net_exp_with_scale_factor")
    bbox_pred = [act(r) for r in bbox_pred]
    if len(bbox_pred) == 1:
        bbox_pred = bbox_pred[0]
    conf_pred = None
    if centerness:
        conf_pred = CenternessNet(n_anchor, use_bias,
                                  convolution = conf_convolution, normalize = conf_normalize, activation = conf_activation,
                                  concat = False, name = "centerness_net")(logits_feature)
    points = generate_points(feature, image_shape, stride = None, normalize = True, concat = False, dtype = tf.float32) #stride = None > Auto Stride (ex: level 3~5 + pooling 6~7 > [8, 16, 32, 64, 128], level 2~5 + pooling 6 > [4, 8, 16, 32, 64])
    result = [r for r in [y_pred, bbox_pred, points, conf_pred] if r is not None]
    return result