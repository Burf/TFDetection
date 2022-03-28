#https://github.com/Burf/ResNet-Tensorflow2
import tensorflow as tf

def normalize(axis = -1, momentum = 0.9, epsilon = 1e-5, **kwargs):
    return tf.keras.layers.BatchNormalization(axis = axis, momentum = momentum, epsilon = epsilon, **kwargs)

def res_basic_block(x, n_feature, stride_size = 1, normalize = normalize, activation = tf.keras.activations.relu, prefix = "", **kwargs):
    out = tf.keras.layers.Conv2D(n_feature, kernel_size = 3, strides = stride_size, padding = "SAME", use_bias = False, kernel_initializer = "he_normal", name = prefix + "conv1")(x)
    out = normalize(name = prefix + "norm1")(out)
    out = tf.keras.layers.Activation(activation, name = prefix + "act1")(out)
    out = tf.keras.layers.Conv2D(n_feature, kernel_size = 3, padding = "SAME", use_bias = False, kernel_initializer = "he_normal", name = prefix + "conv2")(out)
    out = normalize(name = prefix + "norm2")(out)
    if stride_size != 1 or tf.keras.backend.int_shape(x)[-1] != n_feature: #downsample
        x = tf.keras.layers.Conv2D(n_feature, kernel_size = 1, strides = stride_size, padding = "SAME", use_bias = False, kernel_initializer = "he_normal", name = prefix + "downsample_conv")(x)
        x = normalize(name = prefix + "downsample_norm")(x)
    out = tf.keras.layers.Add(name = prefix + "residual")([out, x])
    out = tf.keras.layers.Activation(activation, name = prefix + "out")(out)
    return out

def res_bottleneck_block(x, n_feature, stride_size = 1, group_size = 1, base_width = 64, expansion = 4, normalize = normalize, activation = tf.keras.activations.relu, prefix = "", **kwargs):
    width = int(n_feature * (base_width / 64.)) * group_size
    out = tf.keras.layers.Conv2D(width, kernel_size = 1, strides = 1, padding = "SAME", use_bias = False, kernel_initializer = "he_normal", name = prefix + "conv1")(x)
    out = normalize(name = prefix + "norm1")(out)
    out = tf.keras.layers.Activation(activation, name = prefix + "act1")(out)
    out = tf.keras.layers.Conv2D(width, kernel_size = 3, strides = stride_size, groups = group_size, padding = "SAME", use_bias = False, kernel_initializer = "he_normal", name = prefix + "conv2")(out)
    out = normalize(name = prefix + "norm2")(out)
    out = tf.keras.layers.Activation(activation, name = prefix + "act2")(out)
    out = tf.keras.layers.Conv2D(n_feature * expansion, kernel_size = 1, strides = 1, padding = "SAME", use_bias = False, kernel_initializer = "he_normal", name = prefix + "conv3")(out)
    out = normalize(name = prefix + "norm3")(out)
    if stride_size != 1 or tf.keras.backend.int_shape(x)[-1] != n_feature * expansion: #downsample
        x = tf.keras.layers.Conv2D(n_feature * expansion, kernel_size = 1, strides = stride_size, padding = "SAME", use_bias = False, kernel_initializer = "he_normal", name = prefix + "downsample_conv")(x)
        x = normalize(name = prefix + "downsample_norm")(x)
    out = tf.keras.layers.Add(name = prefix + "residual")([out, x])
    out = tf.keras.layers.Activation(activation, name = prefix + "out")(out)
    return out

def res_stack(x, n_block, n_feature, stride_size = 1, group_size = 1, base_width = 64, block = res_bottleneck_block, normalize = normalize, activation = tf.keras.activations.relu, prefix = ""):
    out = block(x, n_feature = n_feature, stride_size = stride_size, group_size = group_size, base_width = base_width, normalize = normalize, activation = activation, prefix = prefix + "block1_")
    for index in range(1, n_block):
        out = block(out, n_feature = n_feature, group_size = group_size, base_width = base_width, normalize = normalize, activation = activation, prefix = prefix + "block{0}_".format(index + 1))
    return out

def resnet(x, n_blocks, block, n_class = 1000, include_top = False, group_size = 1, base_width = 64, normalize = normalize, activation = tf.keras.activations.relu):
    #https://github.com/Burf/ResNet-Tensorflow2

    #stem
    out = tf.keras.layers.Conv2D(64, kernel_size = 7, strides = 2, padding = "SAME", use_bias = False, kernel_initializer = "he_normal", name = "stem_conv")(x)
    out = normalize(name = "stem_norm")(out)
    out = tf.keras.layers.Activation(activation, name = "stem_act")(out)
    out = tf.keras.layers.MaxPool2D(3, strides = 2, padding = "SAME", name = "stem_pooling")(out)
    
    #stage
    out = res_stack(out, n_blocks[0], 64, group_size = group_size, base_width = base_width, block = block, normalize = normalize, activation = activation, prefix = "stage1_")
    out = res_stack(out, n_blocks[1], 128, stride_size = 2, group_size = group_size, base_width = base_width, block = block, normalize = normalize, activation = activation, prefix = "stage2_")
    out = res_stack(out, n_blocks[2], 256, stride_size = 2, group_size = group_size, base_width = base_width, block = block, normalize = normalize, activation = activation, prefix = "stage3_")
    out = res_stack(out, n_blocks[3], 512, stride_size = 2, group_size = group_size, base_width = base_width, block = block, normalize = normalize, activation = activation, prefix = "stage4_")
    
    #fc
    if include_top:
        out = tf.keras.layers.GlobalAveragePooling2D(name = "global_average_pooling")(out)
        out = tf.keras.layers.Dense(n_class, use_bias = True, kernel_initializer = "he_normal", bias_initializer = "zeros", name = "logits")(out)
    return out

resnet_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}

def load_weight(keras_model, torch_url):
    try:
        import torch
        torch_weight = torch.hub.load_state_dict_from_url(torch_url, map_location = "cpu", progress = True, check_hash = True)
    except:
        print("If you want to use 'ResNet Weight', please install 'torch 1.1▲'")
        return keras_model
    
    conv = []
    bn = {"weight":[], "bias":[], "running_mean":[], "running_var":[]}
    fc = []
    for k, v in dict(torch_weight).items():
        if k.split(".")[-1] in ["weight", "bias", "running_mean", "running_var"]:
            if ("downsample" or "conv" in k) and "weight" in k and v.ndim == 4:
                conv.append(v.permute(2, 3, 1, 0).cpu().data.numpy())
            elif "fc" in k:
                if "weight" in k:
                    v = v.t()
                fc.append(v.cpu().data.numpy())
            else: #bn
                bn[k.split(".")[-1]].append(v.cpu().data.numpy())
    bn = [b for a in [[w, b, m, v] for w, b, m, v in zip(*list(bn.values()))] for b in a]
    
    for w in keras_model.weights:
        if "conv" in w.name:
            new_w = conv.pop(0)
        elif "norm" in w.name:
            new_w = bn.pop(0)
        else: #fc
            new_w = fc.pop(0)
        tf.keras.backend.set_value(w, new_w)
    return keras_model

def resnet18(x, weights = "imagenet"):
    out = resnet(x, [2, 2, 2, 2], res_basic_block, include_top = False, normalize = normalize, activation = tf.keras.activations.relu)
    model = tf.keras.Model(x, out)
    
    if weights == "imagenet":
        load_weight(model, resnet_urls["resnet18"])
    elif weights is not None:
        model.load_weights(weights)
        
    layers = ["stage1_block2_out", "stage2_block2_out", "stage3_block2_out", "stage4_block2_out"]
    return [model.get_layer(l).output for l in layers] 

def resnet34(x, weights = "imagenet"):
    out = resnet(x, [3, 4, 6, 3], res_basic_block, include_top = False, normalize = normalize, activation = tf.keras.activations.relu)
    model = tf.keras.Model(x, out)
    
    if weights == "imagenet":
        load_weight(model, resnet_urls["resnet34"])
    elif weights is not None:
        model.load_weights(weights)
        
    layers = ["stage1_block3_out", "stage2_block4_out", "stage3_block6_out", "stage4_block3_out"]
    return [model.get_layer(l).output for l in layers] 

def resnet50(x, weights = "imagenet"):
    out = resnet(x, [3, 4, 6, 3], res_bottleneck_block, include_top = False, normalize = normalize, activation = tf.keras.activations.relu)
    model = tf.keras.Model(x, out)
    
    if weights == "imagenet":
        load_weight(model, resnet_urls["resnet50"])
    elif weights is not None:
        model.load_weights(weights)
        
    layers = ["stage1_block3_out", "stage2_block4_out", "stage3_block6_out", "stage4_block3_out"]
    return [model.get_layer(l).output for l in layers] 

def resnet101(x, weights = "imagenet"):
    out = resnet(x, [3, 4, 23, 3], res_bottleneck_block, include_top = False, normalize = normalize, activation = tf.keras.activations.relu)
    model = tf.keras.Model(x, out)
    
    if weights == "imagenet":
        load_weight(model, resnet_urls["resnet101"])
    elif weights is not None:
        model.load_weights(weights)
        
    layers = ["stage1_block3_out", "stage2_block4_out", "stage3_block23_out", "stage4_block3_out"]
    return [model.get_layer(l).output for l in layers] 

def resnet152(x, weights = "imagenet"):
    out = resnet(x, [3, 8, 36, 3], res_bottleneck_block, include_top = False, normalize = normalize, activation = tf.keras.activations.relu)
    model = tf.keras.Model(x, out)
    
    if weights == "imagenet":
        load_weight(model, resnet_urls["resnet152"])
    elif weights is not None:
        model.load_weights(weights)
        
    layers = ["stage1_block3_out", "stage2_block8_out", "stage3_block36_out", "stage4_block3_out"]
    return [model.get_layer(l).output for l in layers] 

def resnext50_32x4d(x, weights = "imagenet"):
    out = resnet(x, [3, 4, 6, 3], res_bottleneck_block, include_top = False, group_size = 32, base_width = 4, normalize = normalize, activation = tf.keras.activations.relu)
    model = tf.keras.Model(x, out)
    
    if weights == "imagenet":
        load_weight(model, resnet_urls["resnext50_32x4d"])
    elif weights is not None:
        model.load_weights(weights)
        
    layers = ["stage1_block3_out", "stage2_block4_out", "stage3_block6_out", "stage4_block3_out"]
    return [model.get_layer(l).output for l in layers] 

def resnext101_32x8d(x, weights = "imagenet"):
    out = resnet(x, [3, 4, 23, 3], res_bottleneck_block, include_top = False, group_size = 32, base_width = 8, normalize = normalize, activation = tf.keras.activations.relu)
    model = tf.keras.Model(x, out)

    if weights == "imagenet":
        load_weight(model, resnet_urls["resnext101_32x8d"])
    elif weights is not None:
        model.load_weights(weights)
        
    layers = ["stage1_block3_out", "stage2_block4_out", "stage3_block23_out", "stage4_block3_out"]
    return [model.get_layer(l).output for l in layers] 

def wide_resnet50_2(x, weights = "imagenet"):
    out = resnet(x, [3, 4, 6, 3], res_bottleneck_block, include_top = False, base_width = 128, normalize = normalize, activation = tf.keras.activations.relu)
    model = tf.keras.Model(x, out)
    
    if weights == "imagenet":
        load_weight(model, resnet_urls["wide_resnet50_2"])
    elif weights is not None:
        model.load_weights(weights)
        
    layers = ["stage1_block3_out", "stage2_block4_out", "stage3_block6_out", "stage4_block3_out"]
    return [model.get_layer(l).output for l in layers] 

def wide_resnet101_2(x, weights = "imagenet"):
    out = resnet(x, [3, 4, 23, 3], res_bottleneck_block, include_top = False, base_width = 128, normalize = normalize, activation = tf.keras.activations.relu)
    model = tf.keras.Model(x, out)

    if weights == "imagenet":
        load_weight(model, resnet_urls["wide_resnet101_2"])
    elif weights is not None:
        model.load_weights(weights)
        
    layers = ["stage1_block3_out", "stage2_block4_out", "stage3_block23_out", "stage4_block3_out"]
    return [model.get_layer(l).output for l in layers] 