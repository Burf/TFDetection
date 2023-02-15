import tensorflow as tf
import numpy as np

class PriorProbability(tf.keras.initializers.Initializer):
    """https://github.com/xuannianz/EfficientDet/blob/master/initializers.py"""
    def __init__(self, probability = 0.01):
        self.probability = probability

    def get_config(self):
        return {"probability":self.probability}

    def __call__(self, shape, dtype = None):
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape, dtype = np.float32) * -np.log((1 - self.probability) / self.probability)
        return result