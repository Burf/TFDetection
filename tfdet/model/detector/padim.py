import tensorflow as tf

from tfdet.core.util.distance import mahalanobis
from ..head.padim import FeatureExtractor, Head

def padim(feature, mean, cvar_inv = None, image_shape = [224, 224], sampling_index = None, sigma = 4, metric = mahalanobis, method = "bilinear", memory_reduce = True, batch_size = 1):
    feature = FeatureExtractor(sampling_index = sampling_index, memory_reduce = memory_reduce, name = "feature_extractor")(feature)
    score, mask = Head(mean = mean, cvar_inv = cvar_inv, image_shape = image_shape, sigma = sigma, metric = metric, method = method, batch_size = batch_size, name = "padim")(feature)
    return score, mask