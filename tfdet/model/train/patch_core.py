import tensorflow as tf
import numpy as np

from tfdet.core.ops import core_sampling
from ..head.patch_core import FeatureExtractor

def train(feature, n_sample = 0.001, n_feature = "auto", eps = 0.9):
    if tf.is_tensor(feature):
        b, h, w, c = tf.keras.backend.int_shape(feature)
        feature = tf.reshape(feature, [-1, c])
        feature = tf.py_function(lambda *args: core_sampling(args[0].numpy(), n_sample = n_sample, n_feature = n_feature, eps = eps), inp = [feature], Tout = feature.dtype)
        feature = tf.reshape(feature, [-1, c])
    else:
        b, h, w, c = np.shape(feature)
        feature = np.reshape(feature, [-1, c])
        feature = core_sampling(feature, n_sample = n_sample, n_feature = n_feature, eps = eps)
        feature = np.reshape(feature, [-1, c])
    return feature