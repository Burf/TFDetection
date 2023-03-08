import tensorflow as tf
import numpy as np

from ..head.padim import FeatureExtractor

def decode(fv):
    w = np.identity(np.shape(fv)[-1]) * 0.01
    cvar = np.array([np.cov(f, rowvar = False) for f in fv]) + w
    cvar_inv = np.array([np.linalg.inv(v) for v in cvar])
    return cvar_inv

def train(feature):
    if tf.is_tensor(feature):
        b, h, w, c = tf.keras.backend.int_shape(feature)
        feature = tf.reshape(feature, [-1, h * w, c])
        mean = tf.reduce_mean(feature, axis = 0)
        cvar_inv = tf.py_function(lambda *args: decode(args[0].numpy()), inp = [tf.transpose(feature, [1, 0, 2])], Tout = feature.dtype)
        cvar_inv = tf.reshape(cvar_inv, [h * w, c, c])
    else:
        b, h, w, c = np.shape(feature)
        feature = np.reshape(feature, [-1, h * w, c])
        mean = np.mean(feature, axis = 0)
        cvar_inv = decode(np.transpose(feature, [1, 0, 2]))
        cvar_inv = np.reshape(cvar_inv, [h * w, c, c])
    return mean, cvar_inv