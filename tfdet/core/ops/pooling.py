import tensorflow as tf

class AdaptivePooling(tf.keras.layers.Layer):
    #https://github.com/tensorflow/addons/blob/master/tensorflow_addons/layers/adaptive_pooling.py
    def __init__(self, scale, reduce = tf.reduce_mean, **kwargs):
        super(AdaptivePooling, self).__init__(**kwargs)
        self.scale = scale
        self.reduce = reduce
        
    def build(self, input_shape):
        self.loop = len(input_shape) - 2
        if not isinstance(self.scale, (tuple, list)):
            self.scale = [self.scale] * self.loop
        
    def call(self, inputs):
        out = inputs
        if 0 < self.loop:
            for i in range(self.loop):
                axis = 1 + 2 * i
                out = tf.split(out, self.scale[i], axis = axis)
                out = tf.stack(out, axis = axis)
            out = self.reduce(out, axis = [2 * (i + 1) for i in range(self.loop)])
        return out
    
    def get_config(self):
        config = super(AdaptivePooling, self).get_config()
        config["scale"] = self.scale
        return config
    
class AdaptivePooling2D(AdaptivePooling):
    def __init__(self, scale, reduce = tf.reduce_mean, method = "bilinear", **kwargs):
        super(AdaptivePooling2D, self).__init__(scale = scale, reduce = reduce, **kwargs)
        self.method = method
        
    def call(self, inputs):
        shape = tf.shape(inputs)[1:3]
        w = tf.math.mod(shape, self.scale)
        w = tf.where(w != 0, tf.add(self.scale, -w), w)
        target_size = shape + w
        out = tf.image.resize(inputs, target_size, method = self.method)
        return super(AdaptivePooling2D, self).call(out)
    
    def get_config(self):
        config = super(AdaptivePooling2D, self).get_config()
        config["method"] = self.method
        return config
    
class AdaptiveAveragePooling2D(AdaptivePooling2D):
    def __init__(self, scale, method = "bilinear", **kwargs):
        super(AdaptiveAveragePooling2D, self).__init__(scale = scale, reduce = tf.reduce_mean, method = method, **kwargs)
        
class AdaptiveMaxPooling2D(AdaptivePooling2D):
    def __init__(self, scale, method = "bilinear", **kwargs):
        super(AdaptiveMaxPooling2D, self).__init__(scale = scale, reduce = tf.reduce_max, method = method, **kwargs)     