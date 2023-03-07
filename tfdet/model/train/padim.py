import tensorflow as tf
import numpy as np

from ..head.padim import FeatureExtractor

def train(feature):
    b, h, w, c = tf.keras.backend.int_shape(feature)
    feature = tf.reshape(feature, [-1, h * w, c])
    mean = tf.reduce_mean(feature, axis = 0)
    def func(fv):
        w = np.identity(np.shape(fv)[-1]) * 0.01
        cvar = np.array([np.cov(f, rowvar = False) for f in fv]) + w
        cvar_inv = np.array([np.linalg.inv(v) for v in cvar])
        return cvar_inv
    cvar_inv = tf.py_function(lambda *args: func(args[0].numpy()), inp = [tf.transpose(feature, [1, 0, 2])], Tout = feature.dtype)
    cvar_inv = tf.reshape(cvar_inv, [h * w, c, c])
    return mean, cvar_inv