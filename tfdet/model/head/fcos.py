import tensorflow as tf

from .retina import ClassNet, BoxNet

class CenternessNet(tf.keras.layers.Layer):
    def __init__(self, n_anchor, concat = True, normalize = None, **kwargs):
        super(CenternessNet, self).__init__(**kwargs)
        self.n_anchor = n_anchor
        self.concat = concat
        self.normalize = normalize

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        
        self.layers = [tf.keras.layers.SeparableConv2D(self.n_anchor, 3, padding = "same", depthwise_initializer = tf.keras.initializers.VarianceScaling(), pointwise_initializer = tf.keras.initializers.VarianceScaling(), bias_initializer = "zeros", name = "head")]
        if self.normalize is not None:
            self.layers.append(self.normalize(name = "norm"))
        self.layers.append(tf.keras.layers.Reshape([-1, 1], name = "reshape"))
        self.layers.append(tf.keras.layers.Activation(tf.keras.activations.sigmoid, name = "logits"))
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