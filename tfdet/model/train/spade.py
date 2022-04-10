import tensorflow as tf

from ..head.spade import FeatureExtractor
	
def train(feature, sampling_index = None):
	feature = FeatureExtractor(sampling_index = sampling_index, name = "feature_extractor")(feature)
	return feature