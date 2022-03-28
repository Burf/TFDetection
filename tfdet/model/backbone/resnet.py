import tensorflow as tf

def resnet50(x, weights = "imagenet"):
    model = tf.keras.applications.ResNet50(input_tensor = x, include_top = False, weights = weights)
    layers = ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    return [model.get_layer(l).output for l in layers]
    
def resnet101(x, weights = "imagenet"):
    model = tf.keras.applications.ResNet101(input_tensor = x, include_top = False, weights = weights)
    layers = ["conv2_block3_out", "conv3_block4_out", "conv4_block23_out", "conv5_block3_out"]
    return [model.get_layer(l).output for l in layers]
    
def resnet152(x, weights = "imagenet"):
    model = tf.keras.applications.ResNet152(input_tensor = x, include_top = False, weights = weights)
    layers = ["conv2_block3_out", "conv3_block8_out", "conv4_block36_out", "conv5_block3_out"]
    return [model.get_layer(l).output for l in layers]

def resnet50_v2(x, weights = "imagenet"):
    model = tf.keras.applications.ResNet50V2(input_tensor = x, include_top = False, weights = weights)
    layers = ["conv2_block2_out", "conv3_block3_out", "conv4_block5_out", "conv5_block3_out"]
    return [model.get_layer(l).output for l in layers]
    
def resnet101_v2(x, weights = "imagenet"):
    model = tf.keras.applications.ResNet101V2(input_tensor = x, include_top = False, weights = weights)
    layers = ["conv2_block2_out", "conv3_block3_out", "conv4_block22_out", "conv5_block3_out"]
    return [model.get_layer(l).output for l in layers]
    
def resnet152_v2(x, weights = "imagenet"):
    model = tf.keras.applications.ResNet152V2(input_tensor = x, include_top = False, weights = weights)
    layers = ["conv2_block2_out", "conv3_block7_out", "conv4_block35_out", "conv3_block7_out"]
    return [model.get_layer(l).output for l in layers]