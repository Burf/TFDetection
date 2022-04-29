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