import tensorflow as tf

from ..head.patch_core import FeatureExtractor

def train(feature, sampling_index = None, n_sample = 0.001, n_feature = "auto", eps = 0.9, pool_size = 3, memory_reduce = True):
    try:
        from tfdet.core.util.anodet import core_sampling
        
        feature = FeatureExtractor(sampling_index = sampling_index, pool_size = pool_size, memory_reduce = memory_reduce, name = "feature_extractor")(feature)
        b, h, w, c = tf.keras.backend.int_shape(feature)
        feature = tf.reshape(feature, [-1, c])
        feature = tf.py_function(lambda *args: core_sampling(args[0].numpy(), n_sample = n_sample, n_feature = n_feature, eps = eps), inp = [feature], Tout = feature.dtype)
        feature = tf.reshape(feature, [-1, c])
        return feature
    except:
        print("If you want to use 'PatchCore Train', please install 'scikit-learn 0.13▲'")