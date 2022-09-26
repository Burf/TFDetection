import functools

import numpy as np
import tensorflow as tf

from tfdet.core.util import pipeline, convert_to_numpy, py_func, dict_function

@dict_function(extra_keys = ["y_true", "bbox_true", "mask_true"])
def dict_py_func(function, *args, Tout = tf.float32, **kwargs):
    return py_func(function, *args, Tout = Tout, **kwargs)

def func_format(x_true, y_true = None, bbox_true = None, mask_true = None):
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result 

def pipe(x_true, y_true = None, bbox_true = None, mask_true = None, function = None, dtype = None, tf_func = False,
         batch_size = 0, repeat = 1, shuffle = False, prefetch = False, shuffle_size = None, prefetch_size = None,
         pre_batch_size = 0, pre_unbatch = False, pre_shuffle = False, pre_shuffle_size = None,
         cache = None, num_parallel_calls = None,
         py_func = dict_py_func,
         **kwargs):
    args = [arg for arg in [x_true, y_true, bbox_true, mask_true] if arg is not None]
    args = args[0] if len(args) == 1 else tuple(args)
    func = None
    if callable(function):
        if tf_func:
            func = functools.partial(function, **kwargs)
        else:
            if dtype is None:
                if isinstance(x_true, tf.data.Dataset):
                    sample_args = next(iter(x_true))
                    if isinstance(sample_args, dict):
                        sample_args = convert_to_numpy(sample_args)
                        sample_args = {k:np.stack([v] * pre_batch_size, axis = 0) for k, v in sample_args.items()} if 0 < pre_batch_size else sample_args
                        sample_result = function(**sample_args, **kwargs)
                    else:
                        sample_args = convert_to_numpy([v for v in (sample_args if isinstance(sample_args, tuple) else [sample_args]) if v is not None])
                        sample_args = [np.stack([v] * pre_batch_size, axis = 0) for v in sample_args] if 0 < pre_batch_size else sample_args
                        sample_result = function(*sample_args, **kwargs)
                else:
                    if isinstance(x_true, dict):
                        sample_result = function(**{k:(v[:pre_batch_size] if 0 < pre_batch_size else v[0]) for k, v in x_true.items()}, **kwargs)
                    else:
                        sample_args = [v[:pre_batch_size] if 0 < pre_batch_size else v[0] for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
                        sample_result = function(*sample_args, **kwargs)
                dtype = [tf.convert_to_tensor(r).dtype for r in (sample_result if isinstance(sample_result, tuple) else (sample_result,))]
                dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
            elif np.ndim(dtype) == 0:
                if isinstance(x_true, tf.data.Dataset):
                    if isinstance(x_true.element_spec, tuple) or isinstance(x_true.element_spec, dict):
                        dtype = [dtype] * len(x_true.element_spec)
                else:
                    dtype = [dtype] * len(x_true if isinstance(x_true, dict) else [arg for arg in [x_true, y_true, bbox_true, mask_true] if arg is not None])
                    dtype = dtype[0] if len(dtype) == 1 else tuple(dtype)
            func = functools.partial(py_func, function, Tout = dtype, **kwargs) if callable(function) else None
    return pipeline(args, function = func,
                    batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch, shuffle_size = shuffle_size, prefetch_size = prefetch_size,
                    pre_batch_size = pre_batch_size, pre_unbatch = pre_unbatch, pre_shuffle = pre_shuffle, pre_shuffle_size = pre_shuffle_size,
                    cache = cache, num_parallel_calls = num_parallel_calls)