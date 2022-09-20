import tensorflow as tf

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def separable_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, depthwise_initializer = "he_normal", pointwise_initializer = "he_normal", **kwargs):
    return tf.keras.layers.SeparableConv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, depthwise_initializer = depthwise_initializer, pointwise_initializer = pointwise_initializer, **kwargs)

class WeightedAdd(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WeightedAdd, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name = "weight",
                                 shape = (len(input_shape),),
                                 initializer = tf.keras.initializers.constant(1 / len(input_shape)),
                                 trainable = self.trainable,
                                 dtype = self.dtype)

    def call(self, inputs, **kwargs):
        w = tf.keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis = 0)
        x = x / (tf.reduce_sum(w) + tf.keras.backend.epsilon())
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class FeaturePyramidNetwork(tf.keras.layers.Layer):
    def __init__(self, mode = "bifpn", n_feature = 256, use_bias = True, weighted_add = True, method = "nearest", convolution = separable_conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.nn.swish, **kwargs):
        """
        fpn > mode = "fpn", use_bias = True, weighted_add = False, normalize = None
        panet > mode = "panet", use_bias = True, weighted_add = False, normalize = None
        bifpn > mode = "bifpn", use_bias = True, weighted_add = True, normalize = norm_layer, activation = act_func
        """
        super(FeaturePyramidNetwork, self).__init__(**kwargs)   
        if mode not in ("fpn", "panet", "bifpn"):
            raise ValueError("unknown mode '{0}'".format(mode))
        self.mode = mode
        self.n_feature = n_feature
        self.use_bias = use_bias
        self.weighted_add = weighted_add
        self.method = method
        self.convolution = convolution
        self.normalize = normalize
        self.activation = activation
        self.upsample = tf.keras.layers.Lambda(lambda args: tf.image.resize(args[0], args[1], method = method), name = "upsample")
        
    def build(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]

        self.u2b_resample = []
        if any([s[-1] != self.n_feature for s in input_shape]):
            for index in range(len(input_shape)):
                resample = [tf.keras.layers.Conv2D(self.n_feature, 1, use_bias = self.use_bias, kernel_initializer = "he_normal", name = "{0}_up2bottm_resample_conv{1}".format(self.mode, index + 1))]
                if self.normalize is not None:
                    resample.append(self.normalize(name = "{0}_up2bottm_resample_norm{1}".format(self.mode, index + 1)))
                if self.mode != "bifpn" and self.activation is not None:
                    resample.append(tf.keras.layers.Activation(self.activation, name = "{0}_up2bottom_resample_act{1}".format(self.mode, index + 1)))
                self.u2b_resample.append(resample)
        
        self.u2b_combine = []
        for index in range(1, len(input_shape)):
            #upsample = tf.keras.layers.UpSampling2D(size = (2, 2), interpolation = self.method, name = "{0}_up2bottom_upsample{1}".format(self.mode, index + 1))
            if self.weighted_add:
                add = WeightedAdd(name = "{0}_up2bottom_weighted_add{1}".format(self.mode, index))
            else:
                add = tf.keras.layers.Add(name = "{0}_up2bottom_add{1}".format(self.mode, index))
            combine = []
            if self.mode == "bifpn":
                if self.activation is not None:
                    combine.append(tf.keras.layers.Activation(self.activation, name = "{0}_up2bottom_combine_act{1}".format(self.mode, index)))
                combine.append(self.convolution(self.n_feature, 3, padding = "same", use_bias = self.use_bias, name = "{0}_up2bottom_combine_conv{1}".format(self.mode, index)))
                if self.normalize is not None:
                    combine.append(self.normalize(name = "{0}_up2bottom_combine_norm{1}".format(self.mode, index)))
            #self.u2b_combine.append([upsample, add, combine])
            self.u2b_combine.append([add, combine])
            
        if self.mode != "fpn":
            self.b2u_resample = []
            if any([s[-1] != self.n_feature for s in input_shape]):
                for index in range(len(input_shape)):
                    resample = []
                    if self.mode != "bifpn" or (index != 0 and index != (len(input_shape) - 1)): #For general use, remove (index != len(input_shape) - 2) by pool feature in original implement
                        resample.append(tf.keras.layers.Conv2D(self.n_feature, 1, use_bias = self.use_bias, kernel_initializer = "he_normal", name = "{0}_bottom2up_resample_conv{1}".format(self.mode, index + 1)))
                        if self.normalize is not None:
                            resample.append(self.normalize(name = "{0}_bottom2up_resample_norm{1}".format(self.mode, index + 1)))
                        if self.mode != "bifpn" and self.activation is not None:
                            resample.append(tf.keras.layers.Activation(self.activation, name = "{0}_bottom2up_resample_act{1}".format(self.mode, index + 1)))
                    self.b2u_resample.append(resample)
            
            self.b2u_combine = []
            for index in range(1, len(input_shape)):
                if self.mode == "panet":
                    downsample = [self.convolution(self.n_feature, 3, strides = 2, padding = "same", use_bias = self.use_bias, name = "{0}_bottom2up_downsample_conv{1}".format(self.mode, index))]
                    if self.normalize:
                        downsample.append(self.normalize(name = "{0}_bottom2up_downsample_norm{1}".format(self.mode, index)))
                    if self.activation is not None:
                        downsample.append(tf.keras.layers.Activation(self.activation, name = "{0}_bottom2up_downsample_act{1}".format(self.mode, index)))
                else:
                    downsample = [tf.keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = "same", name = "{0}_bottom2up_downsample{1}".format(self.mode, index))]
                if self.weighted_add:
                    add = WeightedAdd(name = "{0}_bottom2up_weighted_add{1}".format(self.mode, index + 1))
                else:
                    add = tf.keras.layers.Add(name = "{0}_bottom2up_add{1}".format(self.mode, index + 1))
                combine = []
                if self.mode == "bifpn":
                    if self.activation is not None:
                        combine.append(tf.keras.layers.Activation(self.activation, name = "{0}_bottom2up_combine_act{1}".format(self.mode, index + 1)))
                    combine.append(self.convolution(self.n_feature, 3, padding = "same", use_bias = self.use_bias, name = "{0}_bottom2up_combine_conv{1}".format(self.mode, index + 1)))
                    if self.normalize is not None:
                        combine.append(self.normalize(name = "{0}_bottom2up_combine_norm{1}".format(self.mode, index + 1)))
                self.b2u_combine.append([downsample, add, combine])
                
        if self.mode != "bifpn":
            self.post_conv = []
            for index in range(len(input_shape)):
                conv = []
                if self.mode != "panet" or index != 0:
                    conv = [self.convolution(self.n_feature, 3, use_bias = self.use_bias, name = "{0}_post_conv{1}".format(self.mode, index + 1))]
                    if self.normalize is not None:
                        conv.append(self.normalize(name = "{0}_post_norm{1}".format(self.mode, index + 1)))
                    if self.activation is not None:
                        conv.append(tf.keras.layers.Activation(self.activation, name = "{0}_post_act{1}".format(self.mode, index + 1)))
                self.post_conv.append(conv)

    def call(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        feature = list(inputs)
        for i, resample in enumerate(self.u2b_resample):
            for layer in resample:
                feature[i] = layer(feature[i])
        
        u2b = [feature[-1]]
        for index in range(len(feature) - 1, 0, -1):
            #upsample, add, combine = self.u2b_combine[index - 1]
            add, combine = self.u2b_combine[index - 1]
            prev = u2b[-1]
            x = feature[index - 1]
            #out = add([x, upsample(prev)])
            out = add([x, self.upsample([prev, tf.shape(x)[1:3]])])
            for layer in combine:
                out = layer(out)
            u2b.append(out)
        u2b = u2b[::-1]
        
        if self.mode != "fpn":
            feature = u2b if self.mode != "bifpn" else inputs
            for i, resample in enumerate(self.b2u_resample):
                for layer in resample:
                    feature[i] = layer(feature[i])

            b2u = [u2b[0]]
            for index in range(1, len(u2b)):
                downsample, add, combine = self.b2u_combine[index - 1]
                prev = b2u[-1]
                x = u2b[index]
                for layer in downsample:
                    prev = layer(prev)
                args = [x, prev]
                if self.mode == "bifpn" and index != (len(u2b) - 1):
                    args.append(feature[index])
                out = add(args)
                for layer in combine:
                    out = layer(out)
                b2u.append(out)

        out = b2u if self.mode != "fpn" else u2b
        if self.mode != "bifpn":
            for i, conv in enumerate(self.post_conv):
                for layer in conv:
                    out[i] = layer(out[i])
        
        if len(out) == 1:
            out = out[0]
        return out
    
    def get_config(self):
        config = super(FeaturePyramidNetwork, self).get_config()
        config["mode"] = self.mode
        config["n_feature"] = self.n_feature
        config["use_bias"] = self.use_bias
        config["weighted_add"] = self.weighted_add
        config["method"] = self.method
        return config
        
def fpn(n_feature = 256, use_bias = True, weighted_add = False, method = "nearest", convolution = conv, normalize = None, activation = None, mode = "fpn", **kwargs):
    return FeaturePyramidNetwork(mode = mode, n_feature = n_feature, use_bias = use_bias, weighted_add = weighted_add, method = method, convolution = convolution, normalize = normalize, activation = activation, **kwargs)
    
def panet(n_feature = 256, use_bias = True, weighted_add = False, method = "nearest", convolution = conv, normalize = None, activation = None, mode = "panet", **kwargs):
    return FeaturePyramidNetwork(mode = mode, n_feature = n_feature, use_bias = use_bias, weighted_add = weighted_add, method = method, convolution = convolution, normalize = normalize, activation = activation, **kwargs)
    
def bifpn(n_feature = 256, use_bias = True, weighted_add = True, method = "nearest", convolution = separable_conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.nn.swish, mode = "bifpn", **kwargs):
    return FeaturePyramidNetwork(mode = mode, n_feature = n_feature, use_bias = use_bias, weighted_add = weighted_add, method = method, convolution = convolution, normalize = normalize, activation = activation, **kwargs)
    