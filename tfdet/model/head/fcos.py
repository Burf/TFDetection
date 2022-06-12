import tensorflow as tf
import numpy as np

from tfdet.core.anchor import generate_points
from .retina import ClassNet, BoxNet

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

class CenternessNet(tf.keras.layers.Layer):
    def __init__(self, n_anchor, concat = True, convolution = conv, normalize = None, activation = tf.keras.activations.sigmoid, **kwargs):
        super(CenternessNet, self).__init__(**kwargs)
        self.n_anchor = n_anchor
        self.concat = concat
        self.activation = activation
        self.convolution = convolution
        self.normalize = normalize

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        
        self.layers = [self.convolution(self.n_anchor, 3, padding = "same", name = "head")]
        if self.normalize is not None:
            self.layers.append(self.normalize(name = "norm"))
        self.layers.append(tf.keras.layers.Reshape([-1, 1], name = "reshape"))
        self.layers.append(tf.keras.layers.Activation(self.activation, name = "logits"))
        if self.concat and 1 < len(input_shape):
            self.post = tf.keras.layers.Concatenate(axis = -2, name = "logits_concat")

    def call(self, inputs):
        if not isinstance(inputs, list):
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
        return out
    
    def get_config(self):
        config = super(BoxNet, self).get_config()
        config["n_anchor"] = self.n_anchor
        config["concat"] = self.concat
        config["activation"] = self.activation
        config["convolution"] = self.convolution
        config["normalize"] = self.normalize
        return config
    
class Scale(tf.keras.layers.Layer):
    def __init__(self, value = 1., **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.value = value

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        self.w = [self.add_weight(name = "weight{0}".format(index + 1) if 1 < len(input_shape) else "weight",
                                 shape = (1,),
                                 initializer = tf.keras.initializers.constant(self.value),
                                 trainable = self.trainable,
                                 dtype = tf.float32) for index in range(len(input_shape))]

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        out = [inputs[index] * self.w[index] for index in range(len(inputs))]
        if len(out) == 1:
            out = out[0]
        return out

    def get_config(self):
        config = super(Scale, self).get_config()
        config["value"] = self.value
        

def fcos_head(feature, n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4, centerness = True,
              convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu, 
              centerness_convolution = conv, centerness_normalize = None, centerness_activation = tf.keras.activations.sigmoid):
    if tf.is_tensor(image_shape) and 2 < tf.keras.backend.ndim(image_shape) or (not tf.is_tensor(image_shape) and 2 < np.ndim(image_shape)):
        image_shape = tf.shape(image_shape) if tf.keras.backend.int_shape(image_shape)[-3] is None else tf.keras.backend.int_shape(image_shape)
    if 2 < np.shape(image_shape)[0]:
        image_shape = image_shape[-3:-1]
    if not isinstance(feature, list):
        feature = [feature]
    feature = list(feature)
    
    n_anchor = 1
    logits, logits_feature = ClassNet(n_anchor, n_class, n_feature, n_depth, convolution = convolution, normalize = normalize, activation = activation, concat = False, name = "class_net")(feature, feature = True)
    regress = BoxNet(n_anchor, n_feature, n_depth, convolution = convolution, normalize = normalize, activation = activation, concat = False, name = "box_net")(feature)
    regress = Scale(1., name = "box_net_with_scale_factor")(regress)
    if not isinstance(regress, list):
        regress = [regress]
    act = tf.keras.layers.Activation(tf.exp, name = "box_net_exp_with_scale_factor")
    regress = [act(r) for r in regress]
    if len(regress) == 1:
        regress = regress[0]
    if centerness:
        centerness = CenternessNet(n_anchor, concat = False, convolution = centerness_convolution, normalize = centerness_normalize, activation = centerness_activation, name = "centerness_net")(logits_feature)
    else:
        centerness = None
    points = generate_points(feature, image_shape, stride = None, normalize = True, concat = False) #stride = None > Auto Stride (ex: level 3~5 + pooling 6~7 > [8, 16, 32, 64, 128], level 2~5 + pooling 6 > [4, 8, 16, 32, 64])
    result = [r for r in [logits, regress, points, centerness] if r is not None]
    return result