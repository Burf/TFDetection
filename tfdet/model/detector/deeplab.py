import tensorflow as tf

from ..head.deeplab import deeplab_v3 as deeplab_v3_head

def deeplab_v3(x, n_class = 35, rate = [6, 12, 18], n_feature = 256, method = "bilinear", logits_activation = tf.keras.activations.sigmoid, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    if isinstance(x, list):
        x = x[-1]
    return deeplab_v3_head(x, n_class, rate, n_feature, method = method, logits_activation = logits_activation, normalize = normalize, activation = activation)
    
def deeplab_v3_plus(feature, n_class = 35, rate = [6, 12, 18], n_feature = 256, n_low_feature = 48, method = "bilinear", logits_activation = tf.keras.activations.sigmoid, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    if not isinstance(feature, list):
        feature = [feature]
    return deeplab_v3_head(feature, n_class, rate, n_feature, n_low_feature, method = method, logits_activation = logits_activation, normalize = normalize, activation = activation)