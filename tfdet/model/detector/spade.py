import tensorflow as tf

from ..head import spade_head

def spade(feature, feature_vector, image_shape = [224, 224], k = 50, sampling_index = None, sigma = 4, method = "bilinear"):
    score, mask = spade_head(feature, feature_vector, image_shape = image_shape, k = k, sampling_index = sampling_index, sigma = sigma, method = method)
    return score, mask