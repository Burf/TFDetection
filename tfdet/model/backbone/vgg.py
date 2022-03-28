import tensorflow as tf

def vgg16(x, weights = "imagenet"):
    model = tf.keras.applications.VGG16(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2_pool", "block3_pool", "block4_pool", "block5_pool"]
    return [model.get_layer(l).output for l in layers]
    
def vgg19(x, weights = "imagenet"):
    model = tf.keras.applications.VGG19(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2_pool", "block3_pool", "block4_pool", "block5_pool"]
    return [model.get_layer(l).output for l in layers]