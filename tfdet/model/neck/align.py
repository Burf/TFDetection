import tensorflow as tf

from .fpn import fpn

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

class FeatureAlign(tf.keras.layers.Layer):
    def __init__(self, n_feature = 256, n_sampling = 1, pre_sampling = True, neck = fpn, neck_n_depth = 1, convolution = conv, normalize = tf.keras.layers.BatchNormalization, **kwargs):
        super(FeatureAlign, self).__init__(**kwargs)
        self.n_feature = n_feature
        self.n_sampling = n_sampling
        self.pre_sampling = pre_sampling
        self.neck = neck
        self.neck_n_depth = neck_n_depth
        self.convolution = convolution
        self.normalize = normalize
        
    def build(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]  
        
        self.layers = []
        for index in range(self.n_sampling):
            layer = []
            if index == 0:
                layer.append(self.convolution(self.n_feature, 1, use_bias = self.normalize is None))
                if self.normalize is not None:
                    layer.append(self.normalize())
            layer.append(tf.keras.layers.MaxPool2D(3, strides = 2, padding = "same"))
            self.layers.append(layer)
                
        self.neck_layers = []
        if self.neck is not None and 0 < self.neck_n_depth:
            for index in range(self.neck_n_depth):
                self.neck_layers.append(self.neck())
        else:
            self.neck_layers.append([self.convolution(self.n_feature, 1, use_bias = True) for index in range(len(input_shape) + (self.n_sampling if self.pre_sampling else 0))])

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        feature = list(inputs)
        
        if self.pre_sampling:
            x = feature[-1]
            for layers in self.layers:
                for l in layers:
                    x = l(x)
                feature.append(x)
        
        for layer in self.neck_layers:
            if not isinstance(layer, list):
                feature = layer(feature)
            else:
                print(layer)
                for index in range(len(feature)):
                    feature[index] = layer[index](feature[index])
        
        if not isinstance(feature, list):
            feature = [feature]
        if not self.pre_sampling:
            x = feature[-1]
            for layers in self.layers:
                for l in layers:
                    x = l(x)
                feature.append(x)
                
        if len(feature) == 1:
            feature = feature[0]
        return feature

    def get_config(self):
        config = super(FeatureAlign, self).get_config()
        config["n_feature"] = self.n_feature
        config["n_sampling"] = self.n_sampling
        config["pre_sampling"] = self.pre_sampling
        config["neck"] = self.neck
        config["neck_n_depth"] = self.neck_n_depth
        config["convolution"] = self.convolution
        config["normalize"] = self.normalize

class FeatureUpsample(tf.keras.layers.Layer):
    def __init__(self, concat = True, method = "bilinear", **kwargs):
        super(FeatureUpsample, self).__init__(**kwargs)
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
        config = super(FeatureUpsample, self).get_config()
        config["concat"] = self.concat
        config["method"] = self.method
        return config