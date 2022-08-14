import functools

import tensorflow as tf
import numpy as np

from .wrapper import dict_function

def map_fn(function, *args, dtype = tf.float32, batch_size = 1, name = None, **kwargs):
    return tf.map_fn(lambda args: functools.partial(function, **kwargs)(*args), args, dtype = dtype, parallel_iterations = batch_size, name = name)

def convert_to_numpy(*args, return_tuple = False):
    if args and isinstance(args[0], tuple):
        args = args[0]
    args = list(args)
    for index in range(len(args)):
        if tf.is_tensor(args[index]):
            if args[index].dtype == tf.string:
                arg = args[index].numpy()
                if 0 < np.ndim(arg):
                    args[index] = arg.astype(str)
                else:
                    args[index] = arg.decode("UTF-8")
            else:
                args[index] = args[index].numpy()
    if not return_tuple and len(args) == 1:
        args = args[0]
    else:
        args = tuple(args)
    return args

def py_func(function, *args, Tout = tf.float32, **kwargs):
    #return tf.py_function(lambda *args: functools.partial(function, **kwargs)(*convert_to_numpy(*args, return_tuple = True)), args, Tout = Tout)
    tf_kwargs = {k:v for k, v in kwargs.items() if tf.is_tensor(v)}
    return tf.py_function(lambda *args: function(*convert_to_numpy(*args[:-(args[-1] + 1)], return_tuple = True), **dict(kwargs, **{k:v for k,v in zip(list(tf_kwargs), convert_to_numpy(args[-(args[-1] + 1):-1], return_tuple = True))})), inp = args + (*list(tf_kwargs.values()), len(tf_kwargs)), Tout = Tout)

def to_categorical(y, n_class = None, label_smoothing = 0.1):
    result = tf.keras.utils.to_categorical(y, n_class)
    alpha = 1 - label_smoothing
    bias = label_smoothing / (result.shape[-1] - 1)
    return result * (alpha - bias) + bias
    
def pipeline(dataset, map = None, batch_size = 0, epoch = 1, shuffle = False, prefetch = False, post_map = None, num_parallel_calls = None, cache = None, shuffle_size = None, prefetch_size = None):
    if not isinstance(dataset, tf.data.Dataset):
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
    for m in map if isinstance(map, list) else [map]:
        if callable(m):
            dataset = dataset.map(m, num_parallel_calls = num_parallel_calls if num_parallel_calls is not None else tf.data.experimental.AUTOTUNE)
    if isinstance(cache, str):
        dataset = dataset.cache(cache)
    if shuffle:
        dataset = dataset.shuffle(buffer_size = shuffle_size if shuffle_size is not None else max(batch_size, 1) * 10)
    if isinstance(batch_size, int) and 0 < batch_size:
        dataset = dataset.batch(batch_size)
    if isinstance(epoch, int) and 1 < epoch:
        dataset = dataset.repeat(epoch)
    for m in post_map if isinstance(post_map, list) else [post_map]:
        if callable(m):
            dataset = dataset.map(m, num_parallel_calls = num_parallel_calls if num_parallel_calls is not None else tf.data.experimental.AUTOTUNE)
    if prefetch:
        dataset = dataset.prefetch(prefetch_size if prefetch_size is not None else 1)
    return dataset
