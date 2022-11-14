import functools
import os

import tensorflow as tf
import numpy as np

from .wrapper import dict_function

def map_fn(function, *args, dtype = tf.float32, batch_size = 1, name = None, **kwargs):
    func = functools.partial(function, **kwargs)
    batch_shape = [tf.keras.backend.int_shape(arg)[0] for arg in args]
    if np.all([isinstance(b, int) for b in batch_shape]):
        out = []
        for i in range(np.min(batch_shape)):
            x = [arg[i] for arg in args]
            o = func(*x)
            o = (o,) if not isinstance(o, (tuple, list)) else o
            out.append(o)
        out = list(zip(*out))
        out = [tf.stack(o, axis = 0, name = (name if len(out) == 1 else "{0}_{1}".format(name, i + 1)) if name is not None else None) for i, o in enumerate(out)]
        #out = tf.keras.layers.Lambda(lambda args: [tf.stack(arg, axis = 0) for arg in args], name = name)(out)
        if len(out) == 1:
            out = out[0]
        return out
    else:
        #return tf.map_fn(lambda args: functools.partial(function, **kwargs)(*args), args, dtype = dtype, parallel_iterations = batch_size, name = name)
        return tf.map_fn(lambda args: func(*args), args, fn_output_signature = dtype, parallel_iterations = batch_size, name = name)

def convert_to_numpy(*args, return_tuple = False):
    if args and isinstance(args[0], dict):
        return {k:convert_to_numpy(v) for k, v in args[0].items()}
    else:
        if args and (isinstance(args[0], tuple) or isinstance(args[0], list)):
            args = args[0]
        args = list(args)
        for index in range(len(args)):
            arg = args[index]
            if tf.is_tensor(arg):
                dtype, v = arg.dtype, arg.numpy()
                if len(arg.shape) != np.ndim(v):
                    v = [convert_to_numpy(v) for v in arg]
                else:
                    v = [v]
                for i in range(len(v)):
                    if dtype == tf.string and not isinstance(v[i], list):
                        v[i] = v[i].astype(str).astype(np.object0) if 0 < np.ndim(v[i]) else v[i].decode("UTF-8")
                args[index] = v[0] if len(v) == 1 else v
        if not return_tuple and len(args) == 1:
            args = args[0]
        else:
            args = tuple(args)
        return args
        
def convert_to_ragged_tensor(*args, return_tuple = False):
    if args and isinstance(args[0], dict):
        return {k:convert_to_ragged_tensor(v) for k, v in args[0].items()}
    else:
        if args and (isinstance(args[0], tuple) or isinstance(args[0], list)):
            args = args[0]
        args = list(args)
        for index in range(len(args)):
            v = args[index]
            if not isinstance(v, tf.RaggedTensor):
                args[index] = tf.ragged.stack([v])[0]
        if not return_tuple and len(args) == 1:
            args = args[0]
        else:
            args = tuple(args)
        return args
        
def convert_to_tensor(*args, return_tuple = False):
    if args and isinstance(args[0], dict):
        return {k:convert_to_ragged_tensor(v) for k, v in args[0].items()}
    else:
        if args and (isinstance(args[0], tuple) or isinstance(args[0], list)):
            args = args[0]
        args = list(args)
        for index in range(len(args)):
            v = args[index]
            if not tf.is_tensor(v):
                args[index] = tf.convert_to_tensor(v)
            elif isinstance(v, tf.RaggedTensor):
                args[index] = v.to_tensor()
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
    if tf.is_tensor(y):
        if n_class is None:
            n_class = tf.cast(tf.reduce_max(y) + 1, tf.int32)
        result = tf.one_hot(tf.cast(y, tf.int32), n_class)[..., 0, :]
    else:
        result = tf.keras.utils.to_categorical(y, n_class)
    alpha = 1 - label_smoothing
    bias = label_smoothing / (result.shape[-1] - 1)
    return result * (alpha - bias) + bias
    
def pipeline(dataset, function = None,
             batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
             cache = False, num_parallel_calls = None):
    if not isinstance(dataset, tf.data.Dataset):
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
    for func in function if 0 < np.ndim(function) else [function]:
        if callable(func):
            dataset = dataset.map(func, num_parallel_calls = num_parallel_calls if num_parallel_calls is not None else tf.data.experimental.AUTOTUNE)
    if (isinstance(cache, bool) and cache) or isinstance(cache, str):
        dataset = dataset.cache(cache) if isinstance(cache, str) else dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size = shuffle if not isinstance(shuffle, bool) else max(batch_size, 1) * 10)
    if 0 < batch_size:
        dataset = dataset.batch(batch_size)
    if 1 < repeat:
        dataset = dataset.repeat(repeat)
    if prefetch:
        dataset = dataset.prefetch(buffer_size = prefetch if not isinstance(prefetch, bool) else tf.data.experimental.AUTOTUNE)
    return dataset

def zip_pipeline(*args, function = None, num_parallel_calls = 1, **kwargs):
    if len(args) == 1 and 0 < np.ndim(args[0]):
        args = tuple(args[0])
    new_pipe = tf.data.Dataset.zip(args)
    if callable(function):
        def func(*args, function):
            dict_keys = None
            tuple_flag = False
            if isinstance(args[0], dict):
                dict_keys = list(args[0].keys())
                args = [[arg[k] for k in dict_keys] for arg in args]
            elif not isinstance(args[0], (tuple, list)):
                tuple_flag = True
                args = [[arg] for arg in args]
            new_args = [function(arg) for arg in zip(*args)]
            if dict_keys is not None:
                new_args = {k:v for k, v in zip(dict_keys, new_args)}
            elif tuple_flag:
                new_args = new_args[0]
            return new_args
        map_func = functools.partial(func, function = functools.partial(function, **kwargs))
        new_pipe = new_pipe.map(map_func, num_parallel_calls = num_parallel_calls if num_parallel_calls is not None else tf.data.experimental.AUTOTUNE)
    return new_pipe

def concat_pipeline(*args, axis = 0, num_parallel_calls = 1):
    return zip_pipeline(*args, function = tf.concat, axis = axis, num_parallel_calls = num_parallel_calls)

def stack_pipeline(*args, axis = 0, num_parallel_calls = 1):
    return zip_pipeline(*args, function = tf.stack, axis = axis, num_parallel_calls = num_parallel_calls)

def save_model(model, path, graph = True, weight = True, mode = "w"):
    path, ext = os.path.splitext(path)
    ext = ".h5" if len(ext) < 2 else ext
    if graph:
        try:
            with open("{0}{1}".format(path, ".json"), mode = mode) as file:
                file.write(model.to_json())
        except Exception as e:
            print("Failed to save graph : {0}".format(e))
    if weight:
        model.save_weights("{0}{1}".format(path, ext), save_format = ext.replace(".", ""))
    return path

def load_model(path, by_name = False, model = None, weight = True, custom_objects = {}, mode = "r"):
    path, ext = os.path.splitext(path)
    ext = ".h5" if len(ext) < 2 else ext
    if model is None:
        with open("{0}{1}".format(path, ".json"), mode = mode) as file:
            model = tf.keras.models.model_from_json(file.read(), custom_objects)
    if weight:
        try:
            model.load_weights("{0}{1}".format(path, ext), by_name = by_name)
        except Exception as e:
            print("Failed to load weight : {0}".format(e))
    return model

def get_device(type = None):
    try:
        result = tf.config.list_physical_devices(type)
    except:
        from tensorflow.python.client import device_lib
        result = device_lib.list_local_devices()
        if isinstance(type, str):
            result = [device for device in result if device.device_type == type]
    return result

def select_device(gpu = None, limit = None):
    if not isinstance(gpu, list):
        gpu = [gpu]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(device) for device in gpu])
    
    if len(get_device("GPU")) == 0:
        return tf.device("/cpu:0")
    else:
        for device in get_device("GPU"):
            try:
                tf.config.experimental.set_memory_growth(device, True)
                if limit is not None:
                    tf.config.experimental.set_virtual_device_configuration(device, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = limit)])
            except:
                pass
        if len(gpu) == 1:
            device = tf.device("/gpu:0")
        else:
            strategy = tf.distribute.MirroredStrategy(["/gpu:{0}".format(device) for device in range(len(gpu))], cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
            device = strategy.scope()
        return device
    
class EMA:
    """
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    https://github.com/WongKinYiu/yolov7/blob/main/utils/torch_utils.py
    
    1) ema = EMA(model, decay = 0.9999)
    2) update_callback = tf.keras.callbacks.LambdaCallback(on_train_batch_end = lambda step, logs: ema.update() if (step + 1) % 4 == 0 else None)
    3) apply_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end = lambda epoch, logs:ema.apply(), on_epoch_begin = lambda epoch, logs:ema.restroe())
    3) model.fit(...,
                 callbacks=[...,
                            update_callback,
                            apply_callback])
    """
    def __init__(self, model, decay = 0.9999, n_update = 0, ramp = 2000):
        self.model = model
        self.decay = ((lambda x: decay * (1 - np.exp(-x / ramp))) if isinstance(ramp, (int, float)) and ramp != 0 else (lambda x: decay))
        self.n_update = n_update
        
        self.weights = {}
        self.backup = {}
        self.reset()
    
    def reset(self):
        self.weights = {}
        self.backup = {}
        for w in self.model.trainable_weights:
            self.weights[w.name] = np.array(w)
            
    def update(self):
        self.n_update += 1
        decay = self.decay(self.n_update)
        for w in self.model.trainable_weights:
            self.weights[w.name] = (1 - decay) * np.array(w) + (decay * self.weights[w.name])

    def apply(self, model = None):
        for w in (model if isinstance(model, tf.keras.Model) else self.model).trainable_weights:
            self.backup[w.name] = np.array(w)
            tf.keras.backend.set_value(w, self.weights[w.name])
            
    def restore(self, model = None):
        if 0 < len(self.backup):
            for w in (model if isinstance(model, tf.keras.Model) else self.model).trainable_weights:
                if w.name in self.backup:
                    tf.keras.backend.set_value(w, self.backup[w.name])
            self.backup = {}