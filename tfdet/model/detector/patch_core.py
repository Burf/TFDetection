import tensorflow as tf

from ..head.patch_core import FeatureExtractor, Head

def patch_core(feature, feature_vector, image_shape = [224, 224], k = 9, sampling_index = None, pool_size = 3, sigma = 4, method = "bilinear", memory_reduce = True):
    feature = FeatureExtractor(sampling_index = sampling_index, pool_size = pool_size, memory_reduce = memory_reduce, name = "feature_extractor")(feature)
    score, mask = Head(feature_vector = feature_vector, image_shape = image_shape, k = k, sigma = sigma, method = method, name = "patch_core")(feature)
    return score, mask