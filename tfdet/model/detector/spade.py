import tensorflow as tf

from ..head.spade import FeatureExtractor, Head

def spade(feature, feature_vector, image_shape = [224, 224], k = 50, sampling_index = None, sigma = 4, method = "bilinear"):
    feature = FeatureExtractor(sampling_index = sampling_index, name = "feature_extractor")(feature)
    score, mask = Head(feature_vector = feature_vector, image_shape = image_shape, k = k, sigma = sigma, method = method, name = "spade")(feature)
    return score, mask