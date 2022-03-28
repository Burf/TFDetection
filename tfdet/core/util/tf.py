import functools
import tensorflow as tf

def map_fn(function, *args, dtype = tf.float32, batch_size = 1, name = None, **kwargs):
    return tf.map_fn(lambda args: functools.partial(function, **kwargs)(*args), args, dtype = dtype, parallel_iterations = batch_size, name = name)