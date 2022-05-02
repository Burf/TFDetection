import tensorflow as tf

def vgg16(x, weights = "imagenet", indices = None):
    model = tf.keras.applications.VGG16(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2_pool", "block3_pool", "block4_pool", "block5_pool"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature
    
def vgg19(x, weights = "imagenet", indices = None):
    model = tf.keras.applications.VGG19(input_tensor = x, include_top = False, weights = weights)
    layers = ["block2_pool", "block3_pool", "block4_pool", "block5_pool"]
    feature = [model.get_layer(l).output for l in layers]
    if indices is None:
        indices = list(range(len(feature)))
    elif not isinstance(indices, list):
        indices = [indices]
    feature = [feature[index] for index in indices]
    return feature