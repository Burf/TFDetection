import tensorflow as tf
import cv2
import numpy as np
    
class FilterDetection(tf.keras.layers.Layer):
    def __init__(self, threshold, kernel_size = 4, **kwargs):
        super(FilterDetection, self).__init__(**kwargs)
        self.threshold = threshold
        self.kernel_size = kernel_size

    def call(self, inputs):
        score, mask = inputs
        score = tf.where(self.threshold <= score, score, 0)
        mask = tf.where(self.threshold <= mask, mask, 0)
        mask = tf.py_function(lambda *args: np.array([cv2.morphologyEx(o, cv2.MORPH_OPEN, args[1].numpy()) for o in args[0].numpy()]), inp = [mask, tf.ones([self.kernel_size] * 2, dtype = tf.int32)], Tout = mask.dtype)
        mask = tf.reshape(mask, tf.shape(inputs[1]))
        return score, mask
        
    def get_config(self):
        config = super(FilterDetection, self).get_config()
        config["threshold"] = self.threshold
        config["kernel_size"] = self.kernel_size
        return config