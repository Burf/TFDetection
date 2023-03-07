import tensorflow as tf

from ..head import patch_core_head

def patch_core(feature, feature_vector = None, image_shape = [224, 224], k = 9, sampling_index = None, pool_size = 3, sigma = 4, method = "bilinear", memory_reduce = False):
    out = patch_core_head(feature, feature_vector, image_shape = image_shape, k = k, sampling_index = sampling_index, pool_size = pool_size, sigma = sigma, method = method, memory_reduce = memory_reduce)
    return out