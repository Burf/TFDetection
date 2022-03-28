import tensorflow as tf

def mobilenet(x, weights = "imagenet"):
    model = tf.keras.applications.MobileNet(input_tensor = x, include_top = False, weights = weights)
    layers = ["conv_pw_3_relu", "conv_pw_5_relu", "conv_pw_11_relu", "conv_pw_13_relu"]
    return [model.get_layer(l).output for l in layers]
    
def mobilenet_v2(x, weights = "imagenet"):
    model = tf.keras.applications.MobileNetV2(input_tensor = x, include_top = False, weights = weights)
    layers = ["block_2_add", "block_5_add", "block_12_add", "block_16_project_BN"]
    return [model.get_layer(l).output for l in layers]

try:
    def mobilenet_v3_small(x, weights = "imagenet"):
        model = tf.keras.applications.MobileNetV3Small(input_tensor = x, include_top = False, weights = weights)
        layers = ["expanded_conv/project/BatchNorm", "expanded_conv_2/Add", "expanded_conv_7/Add", "expanded_conv_10/Add"]
        return [model.get_layer(l).output for l in layers]

    def mobilenet_v3_large(x, weights = "imagenet"):
        model = tf.keras.applications.MobileNetV3Large(input_tensor = x, include_top = False, weights = weights)
        layers = ["expanded_conv_2/Add", "expanded_conv_5/Add", "expanded_conv_11/Add", "expanded_conv_14/Add"]
        return [model.get_layer(l).output for l in layers]
except:
    print("If you want to use 'MobileNetV3', please install 'tensorflow 2.6▲'")