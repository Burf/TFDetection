import tensorflow as tf

from ..backbone.darknet import darknet_conv_block

def leaky_relu(x, alpha = 0.1):
    return tf.nn.leaky_relu(x, alpha = alpha)
    
def normalize(axis = -1, **kwargs):
    return tf.keras.layers.BatchNormalization(axis = axis, **kwargs)

def yolo_conv_block(x, n_feature, normalize = normalize, activation = leaky_relu):
    out = darknet_conv_block(x, n_feature, 1, normalize = normalize, activation = activation)
    out = darknet_conv_block(out, n_feature * 2, 3, normalize = normalize, activation = activation)
    out = darknet_conv_block(out, n_feature, 1, normalize = normalize, activation = activation)
    out = darknet_conv_block(out, n_feature * 2, 3, normalize = normalize, activation = activation)
    out = darknet_conv_block(out, n_feature, 1, normalize = normalize, activation = activation)
    return out

def yolo_classifier(x, n_class, n_feature, n_anchor = 3, shared = True, normalize = tf.keras.layers.BatchNormalization, activation = leaky_relu):
    out = darknet_conv_block(x, n_feature, 3, normalize = normalize, activation = activation)
    if shared:
        out = darknet_conv_block(out, n_anchor * (n_class + 5), 1, normalize = None, activation = None)
        out = tf.keras.layers.Reshape((-1, (n_class + 5)))(out)
        regress, score, logits = tf.split(out, num_or_size_splits = [4, 1, n_class], axis = -1)
    else:
        score = darknet_conv_block(out, n_anchor * 1, 1, normalize = None, activation = None)
        logits = darknet_conv_block(out, n_anchor * n_class, 1, normalize = None, activation = None)
        regress = darknet_conv_block(out, n_anchor * 4, 1, normalize = None, activation = None)
        score = tf.keras.layers.Reshape((-1, 1))(score)
        logits = tf.keras.layers.Reshape((-1, n_class))(logits)
        regress = tf.keras.layers.Reshape((-1, 4))(regress)
    return [score, logits, regress]