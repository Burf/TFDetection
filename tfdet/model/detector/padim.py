import tensorflow as tf

from tfdet.core.util import mahalanobis
from ..head import padim_head

def padim(feature, mean, cvar_inv = None, image_shape = [224, 224], sampling_index = None, sigma = 4, metric = mahalanobis, method = "bilinear", memory_reduce = True, batch_size = 1):
    score, mask = padim_head(feature, mean, cvar_inv, image_shape = image_shape, sampling_index = sampling_index, sigma = sigma, metric = metric, method = method, memory_reduce = memory_reduce, batch_size = batch_size)
    return score, mask