import tensorflow as tf

from ..backbone.darknet import darknet_conv_block

def leaky_relu(x, alpha = 0.1):
    return tf.nn.leaky_relu(x, alpha = alpha)
    
def yolo_conv_block(x, n_feature, normalize = tf.keras.layers.BatchNormalization, activation = leaky_relu):
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
    xy, wh = tf.split(regress, num_or_size_splits = [2, 2], axis = -1)
    xy = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(xy)
    score = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(score)
    logits = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(logits)
    regress = tf.keras.layers.Concatenate(axis = -1)([xy, wh])
    return [score, logits, regress]