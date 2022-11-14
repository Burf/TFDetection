import functools

import numpy as np
import tensorflow as tf

from tfdet.core.util import (pipeline, convert_to_numpy, convert_to_ragged_tensor, convert_to_tensor, py_func, dict_function,
                             zip_pipeline as zip_pipe, concat_pipeline as concat_pipe, stack_pipeline as stack_pipe)

@dict_function(keys = ["x_true", "y_true", "bbox_true", "mask_true"])
def dict_py_func(function, *args, Tout = tf.float32, **kwargs):
    return py_func(function, *args, Tout = Tout, **kwargs)

@dict_function(keys = ["x_true", "y_true", "bbox_true", "mask_true"])
def dict_tf_func(function, *args, **kwargs):
    return function(*args, **kwargs)

def func_format(x_true, y_true = None, bbox_true = None, mask_true = None):
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result 

def pipe(x_true, y_true = None, bbox_true = None, mask_true = None, function = None,
         batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
         cache = False, num_parallel_calls = None,
         py_func = dict_py_func, tf_func = False, dtype = None,
         **kwargs):
    args = [arg for arg in [x_true, y_true, bbox_true, mask_true] if arg is not None]
    args = args[0] if len(args) == 1 else tuple(args)
    func = None
    if callable(function):
        if tf_func:
            if callable(tf_func):
                func = functools.partial(dict_tf_func, function, **kwargs)
            else:
                func = functools.partial(function, **kwargs)
        else:
            if dtype is None:
                if isinstance(x_true, tf.data.Dataset):
                    sample_args = next(iter(x_true))
                    if isinstance(sample_args, dict):
                        sample_args = convert_to_numpy(sample_args)
                        sample_result = function(**sample_args, **kwargs)
                    else:
                        sample_args = convert_to_numpy([v for v in (sample_args if 0 < np.ndim(sample_args) else [sample_args])])
                        sample_result = function(*sample_args, **kwargs)
                else:
                    if isinstance(x_true, dict):
                        sample_result = function(**{k:v[0] for k, v in x_true.items()}, **kwargs)
                    else:
                        sample_args = [v[0] for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
                        sample_result = function(*sample_args, **kwargs)
                dtype = [tf.convert_to_tensor(r).dtype for r in (sample_result if isinstance(sample_result, tuple) else (sample_result,))]
            elif np.ndim(dtype) == 0:
                if isinstance(x_true, tf.data.Dataset):
                    if isinstance(x_true.element_spec, tuple) or isinstance(x_true.element_spec, dict):
                        dtype = [dtype] * len(x_true.element_spec)
                else:
                    dtype = [dtype] * len(x_true if isinstance(x_true, dict) else [arg for arg in [x_true, y_true, bbox_true, mask_true] if arg is not None])
            if 0 < np.ndim(dtype):        
                dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
            func = functools.partial(py_func, function, Tout = dtype, **kwargs) if callable(function) else None
    return pipeline(args, function = func,
                    batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                    cache = cache, num_parallel_calls = num_parallel_calls)