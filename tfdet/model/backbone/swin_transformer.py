#https://github.com/Burf/SwinTransformer-Tensorflow2
import tensorflow as tf
import numpy as np

def normalize(epsilon = 1e-5, **kwargs):
    return tf.keras.layers.LayerNormalization(epsilon = epsilon, **kwargs)

class Mlp(tf.keras.layers.Layer):
    def __init__(self, n_feature, n_hidden_feature = None, activation = tf.keras.activations.gelu, dropout_rate = 0., **kwargs):
        super(Mlp, self).__init__(**kwargs)
        self.n_feature = n_feature
        self.n_hidden_feature = n_hidden_feature
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.act = tf.keras.layers.Activation(activation, name = "act")
        if 0 < dropout_rate:
            self.dropout = tf.keras.layers.Dropout(dropout_rate, name = "drop")
    
    def build(self, input_shape):
        self.fc1 = tf.keras.layers.Dense(self.n_hidden_feature if self.n_hidden_feature is not None else input_shape[-1], name = "fc1")
        self.fc2 = tf.keras.layers.Dense(self.n_feature, name = "fc2")
    
    def call(self, inputs):
        out = self.fc1(inputs)
        out = self.act(out)
        if 0 < self.dropout_rate:
            out = self.dropout(out)
        out = self.fc2(out)
        if 0 < self.dropout_rate:
            out = self.dropout(out)
        return out
    
    def get_config(self):
        config = super(Mlp, self).get_config()
        config["n_feature"] = self.n_feature
        config["n_hidden_feature"] = self.n_hidden_feature
        config["activation"] = self.activation
        config["dropout_rate"] = self.dropout_rate
        return config

def window_partition(x, size):
    h, w, ch = tf.keras.backend.int_shape(x)[-3:]
    out = tf.reshape(x, [-1, h // size, size, w // size, size, ch])
    out = tf.transpose(out, [0, 1, 3, 2, 4, 5])
    window = tf.reshape(out, [-1, size, size, ch])
    return window

def window_reverse(window, h, w):
    size, ch = tf.keras.backend.int_shape(window)[-2:]
    out = tf.reshape(window, [-1, h // size, w // size, size, size, ch])
    out = tf.transpose(out, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(out, shape=[-1, h, w, ch])
    return x

class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, n_head, window_size, scale = None, use_bias = True, dropout_rate = 0., attention_dropout_rate = 0., **kwargs):
        super(WindowAttention, self).__init__(**kwargs)
        self.n_head = n_head
        self.window_size = window_size if not isinstance(window_size, int) else [window_size, window_size]
        self.scale = scale
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        
        if 0 < dropout_rate:
            self.dropout = tf.keras.layers.Dropout(dropout_rate, name = "proj_drop")
        if 0 < attention_dropout_rate:
            self.attention_dropout = tf.keras.layers.Dropout(attention_dropout_rate, name = "attn_drop")
        
    def build(self, input_shape):
        if self.scale is None:
            self.scale = (input_shape[-1] // self.n_head) ** -0.5
        self.relative_position_bias_table = self.add_weight("relative_position_bias_table", shape = [((2 * self.window_size[0]) - 1) * ((2 * self.window_size[1]) - 1), self.n_head], trainable = self.trainable)
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing = "ij")) #2, Wh, Ww
        coords = np.reshape(coords, [2, -1])
        relative_coords = np.expand_dims(coords, axis = -1) - np.expand_dims(coords, axis = -2) #2, Wh * Ww, Wh * Ww
        relative_coords = np.transpose(relative_coords, [1, 2, 0]) #Wh * Ww, Wh * Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1 #shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = np.sum(relative_coords, -1)
        self.relative_position_index = tf.Variable(tf.convert_to_tensor(relative_position_index), trainable = False, name= "relative_position_index")
        self.qkv = tf.keras.layers.Dense(input_shape[-1] * 3, use_bias = self.use_bias, name = "qkv")
        self.project = tf.keras.layers.Dense(input_shape[-1], use_bias = True, name = "proj")
    
    def call(self, inputs, mask = None):
        """
        x = input features with shape of (num_windows * b, n, ch)
        mask = (0/-inf) mask with shape of (num_windows, Wh * Ww, Wh * Ww) or None
        """
        n, ch = tf.keras.backend.int_shape(inputs)[-2:]
        qkv = tf.transpose(tf.reshape(self.qkv(inputs), [-1, n, 3, self.n_head, ch // self.n_head]), [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ tf.transpose(k, [0, 1, 3, 2]))
        
        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, [-1]))
        relative_position_bias = tf.reshape(relative_position_bias, [self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1]) #Wh * Ww,Wh * Ww, nH
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1]) #nH, Wh * Ww, Wh * Ww
        attn = attn + tf.expand_dims(relative_position_bias, axis = 0)
        
        if mask is not None:
            n_window = tf.keras.backend.int_shape(mask)[0]
            attn = tf.reshape(attn, [-1, n_window, self.n_head, n, n]) + tf.cast(tf.expand_dims(tf.expand_dims(mask, axis = 1), axis = 0), attn.dtype)
            attn = tf.reshape(attn, [-1, self.n_head, n, n])
        attn = tf.nn.softmax(attn, axis = -1)
        
        if 0 < self.attention_dropout_rate:
            attn = self.attention_dropout(attn)

        out = tf.reshape(tf.transpose((attn @ v), [0, 2, 1, 3]), shape=[-1, n, ch])
        out = self.project(out)
        if 0 < self.dropout_rate:
            out = self.dropout(out)
        return out
        
    def get_config(self):
        config = super(WindowAttention, self).get_config()
        config["n_head"] = self.n_head
        config["window_size"] = self.window_size
        config["scale"] = self.scale
        config["use_bias"] = self.use_bias
        config["dropout_rate"] = self.dropout_rate
        config["attention_dropout_rate"] = self.attention_dropout_rate
        return config
    
class DropPath(tf.keras.layers.Layer):
    def __init__(self, rate = 0., **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.rate = rate
        
    def drop_path(self, x, rate):
        keep_prob = 1 - rate
        shape = tf.shape(x)
        w = keep_prob + tf.random.uniform([shape[0]] +  [1] * (len(shape) - 1), dtype = x.dtype)
        return tf.math.divide(x, keep_prob) * tf.floor(w)

    def call(self, inputs):
        out = inputs
        if 0 < self.rate and self.trainable:
            out = self.drop_path(inputs, self.rate)
        return out
    
    def get_config(self):
        config = super(DropPath, self).get_config()
        config["rate"] = self.rate
        return config
        
class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, image_shape, n_head, window_size, shift_size = 0, ratio = 4., scale = None, use_bias = True, dropout_rate = 0., attention_dropout_rate = 0., droppath_rate = 0., normalize = normalize, activation = tf.keras.activations.gelu, **kwargs):
        super(SwinTransformerBlock, self).__init__(**kwargs)
        self.image_shape = image_shape
        self.n_head = n_head
        self.window_size = window_size
        self.shift_size = shift_size
        self.ratio = ratio
        if min(image_shape) <= window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.window_size = min(image_shape)
            self.shift_size = 0
        assert 0 <= shift_size < window_size, "shift_size must in 0-window_size"
        self.scale = scale
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.droppath_rate = droppath_rate
        self.normalize = normalize
        self.activation = activation
        
        self.norm1 = normalize(name = "norm1")
        self.attn = WindowAttention(n_head, window_size, scale, use_bias, dropout_rate, attention_dropout_rate, name = "attn")
        if 0 < droppath_rate:
            self.droppath = DropPath(droppath_rate, name = "drop_path")
        self.norm2 = normalize(name = "norm2")
        
    def build(self, input_shape):
        self.mlp = Mlp(input_shape[-1], int(input_shape[-1] * self.ratio), self.activation, self.dropout_rate, name = "mlp")
        self.attn_mask = None
        if 0 < self.shift_size:
            h, w = self.image_shape
            mask = np.zeros([1, h, w, 1])
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for hs in h_slices:
                for ws in w_slices:
                    mask[:, hs, ws, :] = cnt
                    cnt += 1

            mask_windows = window_partition(tf.convert_to_tensor(mask), self.window_size)
            mask_windows = tf.reshape(mask_windows, [-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(mask_windows, axis = 1) - tf.expand_dims(mask_windows, axis = 2)
            attn_mask = tf.where(attn_mask != 0, -100, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0, attn_mask)
            self.attn_mask = tf.Variable(attn_mask, trainable = False, name = "attn_mask")
    
    def call(self, inputs):
        h, w = self.image_shape
        hw, ch = tf.keras.backend.int_shape(inputs)[-2:]
        assert h * w == hw, "input feature has wrong size"
        
        out = self.norm1(inputs)
        out = tf.reshape(out, shape=[-1, h, w, ch])
        
        #cyclic shift
        if 0 < self.shift_size:
            out = tf.roll(out, [-self.shift_size, -self.shift_size], axis = [1, 2])
        
        #partition windows
        out = window_partition(out, self.window_size)
        out = tf.reshape(out, [-1, self.window_size * self.window_size, ch])
        
        #W-MSA/SW-MSA
        out = self.attn(out, mask = self.attn_mask)
        
        #merge windows
        out = tf.reshape(out, shape=[-1, self.window_size, self.window_size, ch])
        out = window_reverse(out, h, w)
        
        #reverse cyclic shift
        if 0 < self.shift_size:
            out = tf.roll(out, [self.shift_size, self.shift_size], axis = [1, 2])
        out = tf.reshape(out, [-1, hw, ch])
        
        #FFN
        if 0 < self.droppath_rate:
            out = self.droppath(out)
        shortcut = out = inputs + out
        out = self.mlp(self.norm2(out))
        if 0 < self.droppath_rate:
            out = self.droppath(out)
        out = shortcut + out
        return out
        
    def get_config(self):
        config = super(SwinTransformerBlock, self).get_config()
        config["image_shape"] = self.image_shape
        config["n_head"] = self.n_head
        config["window_size"] = self.window_size
        config["shift_size"] = self.shift_size
        config["ratio"] = self.ratio
        config["scale"] = self.scale
        config["use_bias"] = self.use_bias
        config["dropout_rate"] = self.dropout_rate
        config["attention_dropout_rate"] = self.attention_dropout_rate
        config["droppath_rate"] = self.droppath_rate
        config["normalize"] = self.normalize
        config["activation"] = self.activation
        return config
    
class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, image_shape, normalize = normalize, **kwargs):
        super(PatchMerging, self).__init__(**kwargs)
        self.image_shape = image_shape
        self.normalize = normalize
        
        self.norm = normalize(name = "norm")
        
    def build(self, input_shape):
        self.reduction = tf.keras.layers.Dense(input_shape[-1] * 2, use_bias = False, name = "reduction")

    def call(self, inputs):
        h, w = self.image_shape
        hw, ch = tf.keras.backend.int_shape(inputs)[-2:]
        assert h * w == hw, "input feature has wrong size"
        assert h % 2 == 0 and w % 2 == 0, "x size ({0} * {1}) are not even.".format(h, w)

        out = tf.reshape(inputs, [-1, h, w, ch])

        o0 = out[:, 0::2, 0::2, :] #b, h / 2, w / 2, ch
        o1 = out[:, 1::2, 0::2, :] #b, h / 2, w / 2, ch
        o2 = out[:, 0::2, 1::2, :] #b, h / 2, w / 2, ch
        o3 = out[:, 1::2, 1::2, :] #b, h / 2, w / 2, ch
        out = tf.concat([o0, o1, o2, o3], axis = -1) #b, h / 2, w / 2, 4 * ch
        out = tf.reshape(out, [-1, (h // 2) * (w // 2), 4 * ch]) #b, h / 2 * w / 2, 4 * ch

        out = self.norm(out)
        out = self.reduction(out)
        return out
        
    def get_config(self):
        config = super(PatchMerging, self).get_config()
        config["image_shape"] = self.image_shape
        config["normalize"] = self.normalize
        return config
    
class BasicLayer(tf.keras.layers.Layer):
    def __init__(self, image_shape, n_block, n_head, window_size, ratio = 4., scale = None, use_bias = True, dropout_rate = 0., attention_dropout_rate = 0., droppath_rate = 0., normalize = normalize, activation = tf.keras.activations.gelu, downsample = None, **kwargs):
        super(BasicLayer, self).__init__(**kwargs)
        self.image_shape = image_shape
        self.n_block = n_block
        self.n_head = n_head
        self.window_size = window_size
        self.ratio = ratio
        self.scale = scale
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.droppath_rate = droppath_rate
        self.normalize = normalize
        self.activation = activation
        self.downsample = downsample
        
        #build blocks
        self.blocks = [SwinTransformerBlock(image_shape, n_head, window_size,
                                            shift_size = 0 if (i % 2 == 0) else window_size // 2,
                                            ratio = ratio, scale = scale, use_bias = use_bias,
                                            dropout_rate = dropout_rate, attention_dropout_rate = attention_dropout_rate,
                                            droppath_rate = droppath_rate[i] if not (isinstance(droppath_rate, float) or isinstance(droppath_rate, int)) else droppath_rate,
                                            normalize = normalize, activation = activation, name = "blocks_{0}".format(i))
                       for i in range(n_block)]
        
        #patch merging block
        if downsample is not None:
            self.ds = downsample(image_shape, normalize, name = "downsample")
    
    def call(self, inputs):
        out = inputs
        for block in self.blocks:
            out = block(out)
        if self.downsample is not None:
            out = self.ds(out)
        return out
        
    def get_config(self):
        config = super(BasicLayer, self).get_config()
        config["image_shape"] = self.image_shape
        config["n_block"] = self.n_block
        config["n_head"] = self.n_head
        config["window_size"] = self.window_size
        config["ratio"] = self.ratio
        config["scale"] = self.scale
        config["use_bias"] = self.use_bias
        config["dropout_rate"] = self.dropout_rate
        config["attention_dropout_rate"] = self.attention_dropout_rate
        config["droppath_rate"] = self.droppath_rate
        config["normalize"] = self.normalize
        config["activation"] = self.activation
        config["downsample"] = self.downsample
        return config
    
class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, patch_size = 4, n_feature = 96, normalize = None, **kwargs):
        super(PatchEmbed, self).__init__(**kwargs)
        self.patch_size = patch_size if not isinstance(patch_size, int) else [patch_size, patch_size]
        self.n_feature = n_feature
        self.normalize = normalize
        
        self.project = tf.keras.layers.Conv2D(n_feature, patch_size, strides = patch_size, use_bias = True, name = "proj")
        if normalize is not None:
            self.norm = normalize(name = "norm")
        
    def call(self, inputs):
        out = self.project(inputs)
        out = tf.reshape(out, [-1, tf.shape(out)[-3] * tf.shape(out)[-2], self.n_feature])
        if self.normalize is not None:
            out = self.norm(out)
        return out
        
    def get_config(self):
        config = super(PatchEmbed, self).get_config()
        config["patch_size"] = self.patch_size
        config["n_feature"] = self.n_feature
        config["normalize"] = self.normalize
        return config
    
def swin_transformer(x, n_class = 1000, include_top = True, patch_size = 4, n_feature = 96, n_blocks = [2, 2, 6, 2], n_heads = [3, 6, 12, 24], window_size = 7, ratio = 4., scale = None, use_bias = True, patch_normalize = True, dropout_rate = 0., attention_dropout_rate = 0., droppath_rate = 0.1, normalize = normalize, activation = tf.keras.activations.gelu):
    """
    https://github.com/Burf/SwinTransformer-Tensorflow2
    input_shape 224, 224, 3 > window_size 7, input_shape 384, 384, 3 > window_size 
    """
    patch_size = patch_size if not isinstance(patch_size, int) else [patch_size, patch_size]
    h, w = tf.keras.backend.int_shape(x)[-3:-1]
    patch_shape = [h // patch_size[0], w // patch_size[1]]
    
    droppath_rate = np.linspace(0., droppath_rate, sum(n_blocks))
    
    out = PatchEmbed(patch_size, n_feature, normalize if patch_normalize else None, name = "patch_embed")(x)
    if 0 < dropout_rate:
        out = tf.keras.layers.Dropout(dropout_rate, name = "pos_drop")(out)
    for i in range(len(n_blocks)):
        out = BasicLayer([patch_shape[0] // (2 ** i), patch_shape[1] // (2 ** i)], n_blocks[i], n_heads[i], window_size, ratio, scale, use_bias, dropout_rate, attention_dropout_rate, droppath_rate[sum(n_blocks[:i]):sum(n_blocks[:i + 1])], normalize, activation, downsample = PatchMerging if i < (len(n_blocks) - 1) else None, name = "layers_{0}".format(i))(out)
    out = normalize(name = "norm")(out) #b, hw, c
    
    if include_top:
        out = tf.keras.layers.GlobalAveragePooling1D(name = "avgpool")(out)
        out = tf.keras.layers.Dense(n_class, name = "head")(out)
    return out

swin_transformer_url = {
    "swin_tiny_window7_224":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
    "swin_small_window7_224":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
    "swin_base_window7_224":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth",
    "swin_base_window12_384":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth",
    "swin_base_window7_224_22kto1k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth",
    "swin_base_window12_384_22kto1k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth",
    "swin_large_window7_224_22kto1k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth",
    "swin_large_window12_384_22kto1k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth",
    "swin_base_window7_224_22k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth",
    "swin_base_window12_384_22k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth",
    "swin_large_window7_224_22k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth",
    "swin_large_window12_384_22k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth"
}

def load_weight(keras_model, torch_url):
    """
    https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
    https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth
    https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth
    https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth
    https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth
    https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
    https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
    https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
    https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
    """
    try:
        import torch
        torch_weight = torch.hub.load_state_dict_from_url(torch_url, map_location = "cpu", progress = True, check_hash = True)
    except:
        print("If you want to use 'SwinTransformerV1 Weight', please install 'torch 1.1▲'")
        return keras_model
    
    weight = {}
    for k, v in torch_weight["model"].items():
        names = k.split(".")
        if "weight" == names[-1]:
            if v.ndim == 4:
                v = v.permute(2, 3, 1, 0)
            else:
                v = v.t()
        weight["_".join(names)] = v.cpu().data.numpy()
    
    keys = [w.name.split(":")[0].replace("/", "_").replace("kernel", "weight").replace("gamma", "weight").replace("beta", "bias") for w in keras_model.weights]
    values = [weight[k] for k in keys]
    
    for w in keras_model.weights:
        try:
            tf.keras.backend.set_value(w, values.pop(0))
        except:
            pass
    return keras_model
    
def get_shape(size, input_shape = None):
    if input_shape is None:
        input_shape = [1, 1]
    max_index = np.argmax(input_shape)
    scale = (np.max(input_shape) / np.min(input_shape))
    size = np.sqrt(size / scale)
    shape = [int(size), int(size * scale)]
    if max_index == 0:
        shape = shape[::-1]
    return shape

def swin_transformer_tiny(x, patch_size = 4, n_feature = 96, n_blocks = [2, 2, 6, 2], n_heads = [3, 6, 12, 24], window_size = 7, ratio = 4., scale = None, use_bias = True, patch_normalize = True, dropout_rate = 0., attention_dropout_rate = 0., droppath_rate = 0.1, normalize = normalize, activation = tf.keras.activations.gelu, weights = "imagenet", indices = None):
    input_shape = tf.keras.backend.int_shape(x)[-3:-1]
    
    out = swin_transformer(x, include_top = False, patch_size = patch_size, n_feature = n_feature, n_blocks = n_blocks, n_heads = n_heads, window_size = window_size, ratio = ratio, scale = scale, use_bias = use_bias, patch_normalize = patch_normalize, dropout_rate = dropout_rate, attention_dropout_rate = attention_dropout_rate, droppath_rate = droppath_rate, normalize = normalize, activation = activation)
    model = tf.keras.Model(x, out)
    layers = ["layers_{0}".format(index) for index in range(len(n_blocks) - 2)] + ["layers_{0}".format(len(n_blocks) - 1)]
    
    if weights == "imagenet":
        load_weight(model, swin_transformer_url["swin_tiny_window7_224"])
    elif weights is not None:
        model.load_weights(weights)
    
    feature = []
    for i, l in enumerate(layers):
        out = model.get_layer(l).output
        out = normalize(name = "{0}_norm".format(l))(out)
        out = tf.keras.layers.Reshape([*get_shape(tf.keras.backend.int_shape(out)[-2], input_shape), tf.keras.backend.int_shape(out)[-1]], name = "{0}_feature".format(l))(out)
        feature.append(out)
        
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return featurefeature
    
def swin_transformer_small(x, patch_size = 4, n_feature = 96, n_blocks = [2, 2, 18, 2], n_heads = [3, 6, 12, 24], window_size = 7, ratio = 4., scale = None, use_bias = True, patch_normalize = True, dropout_rate = 0., attention_dropout_rate = 0., droppath_rate = 0.1, normalize = normalize, activation = tf.keras.activations.gelu, weights = "imagenet", indices = None):
    input_shape = tf.keras.backend.int_shape(x)[-3:-1]
    
    out = swin_transformer(x, include_top = False, patch_size = patch_size, n_feature = n_feature, n_blocks = n_blocks, n_heads = n_heads, window_size = window_size, ratio = ratio, scale = scale, use_bias = use_bias, patch_normalize = patch_normalize, dropout_rate = dropout_rate, attention_dropout_rate = attention_dropout_rate, droppath_rate = droppath_rate, normalize = normalize, activation = activation)
    model = tf.keras.Model(x, out)
    layers = ["layers_{0}".format(index) for index in range(len(n_blocks) - 2)] + ["layers_{0}".format(len(n_blocks) - 1)]
    
    if weights == "imagenet":
        load_weight(model, swin_transformer_url["swin_small_window7_224"])
    elif weights is not None:
        model.load_weights(weights)
    
    feature = []
    for i, l in enumerate(layers):
        out = model.get_layer(l).output
        out = normalize(name = "{0}_norm".format(l))(out)
        out = tf.keras.layers.Reshape([*get_shape(tf.keras.backend.int_shape(out)[-2], input_shape), tf.keras.backend.int_shape(out)[-1]], name = "{0}_feature".format(l))(out)
        feature.append(out)
        
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def swin_transformer_base(x, patch_size = 4, n_feature = 128, n_blocks = [2, 2, 18, 2], n_heads = [4, 8, 16, 32], window_size = 7, ratio = 4., scale = None, use_bias = True, patch_normalize = True, dropout_rate = 0., attention_dropout_rate = 0., droppath_rate = 0.1, normalize = normalize, activation = tf.keras.activations.gelu, weights = "imagenet_22k", indices = None):
    #input_shape 224, 224, 3 > window_size 7, input_shape 384, 384, 3 > window_size 12
    input_shape = tf.keras.backend.int_shape(x)[-3:-1]
    w_size = 7
    image_size = 224
    if 384 <= np.min(input_shape):
        w_size = 12
        image_size = 384

    out = swin_transformer(x, include_top = False, patch_size = patch_size, n_feature = n_feature, n_blocks = n_blocks, n_heads = n_heads, window_size = window_size, ratio = ratio, scale = scale, use_bias = use_bias, patch_normalize = patch_normalize, dropout_rate = dropout_rate, attention_dropout_rate = attention_dropout_rate, droppath_rate = droppath_rate, normalize = normalize, activation = activation)
    model = tf.keras.Model(x, out)
    layers = ["layers_{0}".format(index) for index in range(len(n_blocks) - 2)] + ["layers_{0}".format(len(n_blocks) - 1)]
     
    if weights == "imagenet":
        load_weight(model, swin_transformer_url["swin_base_window{0}_{1}".format(w_size, image_size)])
    elif weights == "imagenet_22kto1k":
        load_weight(model, swin_transformer_url["swin_base_window{0}_{1}_22kto1k".format(w_size, image_size)])
    elif weights == "imagenet_22k":
        load_weight(model, swin_transformer_url["swin_base_window{0}_{1}_22k".format(w_size, image_size)])
    elif weights is not None:
        model.load_weights(weights)
    
    feature = []
    for i, l in enumerate(layers):
        out = model.get_layer(l).output
        out = normalize(name = "{0}_norm".format(l))(out)
        out = tf.keras.layers.Reshape([*get_shape(tf.keras.backend.int_shape(out)[-2], input_shape), tf.keras.backend.int_shape(out)[-1]], name = "{0}_feature".format(l))(out)
        feature.append(out)
        
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature

def swin_transformer_large(x, patch_size = 4, n_feature = 192, n_blocks = [2, 2, 18, 2], n_heads = [6, 12, 24, 48], window_size = 7, ratio = 4., scale = None, use_bias = True, patch_normalize = True, dropout_rate = 0., attention_dropout_rate = 0., droppath_rate = 0.1, normalize = normalize, activation = tf.keras.activations.gelu, weights = "imagenet_22k", indices = None):
    #input_shape 224, 224, 3 > window_size 7, input_shape 384, 384, 3 > window_size 12
    input_shape = tf.keras.backend.int_shape(x)[-3:-1]
    w_size = 7
    image_size = 224
    if 384 <= np.min(input_shape):
        w_size = 12
        image_size = 384

    out = swin_transformer(x, include_top = False, patch_size = patch_size, n_feature = n_feature, n_blocks = n_blocks, n_heads = n_heads, window_size = window_size, ratio = ratio, scale = scale, use_bias = use_bias, patch_normalize = patch_normalize, dropout_rate = dropout_rate, attention_dropout_rate = attention_dropout_rate, droppath_rate = droppath_rate, normalize = normalize, activation = activation)
    model = tf.keras.Model(x, out)
    layers = ["layers_{0}".format(index) for index in range(len(n_blocks) - 2)] + ["layers_{0}".format(len(n_blocks) - 1)]
    
    if weights == "imagenet_22kto1k":
        load_weight(model, swin_transformer_url["swin_large_window{0}_{1}_22kto1k".format(w_size, image_size)])
    elif weights == "imagenet_22k":
        load_weight(model, swin_transformer_url["swin_large_window{0}_{1}_22k".format(w_size, image_size)])
    elif weights is not None:
        model.load_weights(weights)
    
    feature = []
    for i, l in enumerate(layers):
        out = model.get_layer(l).output
        out = normalize(name = "{0}_norm".format(l))(out)
        out = tf.keras.layers.Reshape([*get_shape(tf.keras.backend.int_shape(out)[-2], input_shape), tf.keras.backend.int_shape(out)[-1]], name = "{0}_feature".format(l))(out)
        feature.append(out)
        
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature