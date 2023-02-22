import functools
import os
import inspect
from multiprocessing.pool import ThreadPool

import numpy as np
import tensorflow as tf

from tfdet.builder import build_transform
from tfdet.core.util import dict_function, py_func, pipeline
from tfdet.dataset.util import save_pickle, load_pickle

def multi_transform(function = None, sample_size = None):
    def wrapper(function):
        if sample_size is not None:
            function.sample_size = sample_size
        return function
    if function is not None:
        if callable(function):
            return wrapper(function)
        else:
            sample_size = function
            function = None
    return wrapper

class Dataset:
    def __init__(self, *args, transform = None, preprocess = None, shuffle = False, cache = None, keys = ["x_true", "y_true", "bbox_true", "mask_true"]):
        """
        args > x_true, y_true, bbox_true, mask_true(optional) style args or custom args(should change keys) or dataset
        transform or preprocess > {'name':transform name or func, **kwargs} or transform name or func #find module in tfdet.dataset.transform and map kwargs.
                                  kwargs["sample_size"] > Covnert transform into multi_transform.(If transform doesn't need sample_size.)
        
        <example>
        1. basic
        > dataset = tfdet.dataset.Dataset(x_true, y_true, bbox_true, mask_true,
                                          transform = [{"name":"load"},
                                                       {"name":"resize", "image_shape":[512, 512]},
                                                       {"name":"pad", "image_shape":[512, 512]},
                                                       {"name":"filter_annotation"},
                                                       {"name":"label_encode", "label":tfdet.dataset.coco.LABEL},
                                                       {"name":"normalize", "mean":[123.675, 116.28, 103.53], "std":[58.395, 57.12, 57.375]}], #post-apply transform
                                          preprocess = [], #pre-apply transform
                                          shuffle = False, #when item 0 is called, shuffle indices.(Recommended by 1 GPU)
                                          cache = "dataset.cache", #save cache after preprocess
                                          keys = ["x_true", "y_true", "bbox_true", "mask_true"]) #transform mapping keys for args
        > dataset[i] #or next(iter(dataset))
        
        2. dataset
        > dataset = tfdet.dataset.coco.load_dataset("./coco/annotations/instances_train2017.json", "./coco/train2017",
                                                    mask = False, crowd = False,
                                                    cache = "coco_train.cache")
        > dataset = tfdet.dataset.Dataset(dataset,
                                          transform = [{"name":"load"},
                                                       {"name":"resize", "image_shape":[512, 512]},
                                                       {"name":"pad", "image_shape":[512, 512]},
                                                       {"name":"filter_annotation"},
                                                       {"name":"label_encode", "label":tfdet.dataset.coco.LABEL},
                                                       {"name":"normalize", "mean":[123.675, 116.28, 103.53], "std":[58.395, 57.12, 57.375]}])
        > dataset[i] #or next(iter(dataset))
        
        3. dataset to pipe
        > pipe = tfdet.dataset.PipeLoader(dataset)
        > pipe = tfdet.dataset.pipeline.args2dict(pipe) #for train_model
        > pipe = tfdet.dataset.pipeline.collect(pipe) #filtered item by key
        > pipe = tfdet.dataset.pipeline.cast(pipe)
        > pipe = tfdet.dataset.pipeline.key_map(pipe, batch_size = 16, shuffle = False, prefetch = True)
        > next(iter(dataset))
        """
        self.args = args
        
        transform = build_transform(transform, key = "name")
        preprocess = build_transform(preprocess, key = "name")
        self.transform = [transform] if not isinstance(transform, (list, tuple)) else transform
        self.preprocess = [preprocess] if not isinstance(preprocess, (list, tuple)) else preprocess
        self.shuffle = shuffle
        self.cache = cache
        self.keys = keys
        
        self.prepare()

    def set_length(self):
        try:
            if isinstance(self.args[0], Dataset):
                length = len(self.args[0])
            elif isinstance(self.args[0], dict):
                length = len(list(self.args[0].values())[0])
            else:
                length = len(self.args[0])
        except:
            length = -1
        self.length = length
    
    def __len__(self):
        return self.length
    
    def set_indices(self, shuffle = False):
        indices = np.arange(self.length)
        if shuffle:
            np.random.shuffle(indices)
        self.indices = indices
    
    @staticmethod
    def slice(*args, indices):
        multi = np.ndim(indices) != 0
        if isinstance(args[0], dict):
            args = ({k:v[indices] if isinstance(v, np.ndarray) else ([v[ind] for ind in indices] if multi else v[indices]) for k, v in args[0].items()},)
        else:
            args = tuple([v[indices] if isinstance(v, np.ndarray) else ([v[ind] for ind in indices] if multi else v[indices]) for v in args])
        return args
    
    @staticmethod
    def stack(*args):
        args = [(arg,) if not isinstance(arg, tuple) else arg for arg in args]
        if isinstance(args[0][0], dict):
            result = None
            for arg in args:
                if result is None:
                    result = {k:[v] for k, v in arg[0].items()}
                else:
                    for k in result.keys():
                        result[k].append(arg[0][k])
            args = (result,)#({k:np.array(v) for k, v in result.items()},)
        else:
            args = tuple([list(arg) for arg in zip(*args)])#tuple([np.array(arg) for arg in zip(*args)])
        return args
    
    def get(self, index, transform = None):
        if transform is None:
            transform = self.transform
        elif not isinstance(transform, (list, tuple)):
            transform = [transform]
        else:
            pass
        
        if isinstance(self.args[0], Dataset):
            item = self.args[0][index]
            if not isinstance(item, tuple):
                item = (item,)
        else:
            item = self.slice(*self.args, indices = index)
        for j in range(len(transform)):
            func = transform[j]
            if callable(func):
                if hasattr(func, "func") and hasattr(func.func, "sample_size"):
                    sample_size = func.func.sample_size
                elif hasattr(func, "sample_size"):
                    sample_size = func.sample_size
                else:
                    sample_size = None
                if sample_size is not None:
                    sample_item = []
                    if 0 < sample_size:
                        sample_indices = np.random.choice(self.indices, sample_size, replace = True)
                        sample_item = [self.get(index, transform = transform[:j]) for index in sample_indices]
                    item = self.stack(item, *sample_item)
                item = dict_function(self.keys)(func)(*item)
                if not isinstance(item, tuple):
                    item = (item,)
        #item = [np.ascontiguousarray(arg) if isinstance(arg, np.ndarray) else arg for arg in item]
        return item[0] if len(item) == 1 else tuple(item)
    
    def prepare(self):
        if isinstance(self.cache, str) and os.path.exists(self.cache):
            args = load_pickle(self.cache)
        else:
            self.set_length()
            self.set_indices()
            args = self.args
            if any([callable(func) for func in self.preprocess]):
                indices = self.indices

                iter_data = ThreadPool(8).imap(lambda index: self.get(index, transform = self.preprocess), indices)
                try:
                    from tqdm import tqdm
                    iter_data = tqdm(iter_data, total = len(indices), desc = "Preprocessing Data")
                except:
                    pass
                
                iter_data = list(iter_data)
                args = self.stack(*iter_data)
                if hasattr(iter_data, "close"):
                    iter_data.close()
            #args = (args,) if not isinstance(args, tuple) else args
            if isinstance(self.cache, str) and not os.path.exists(self.cache):
                save_pickle(args, self.cache)
        self.args = args
        self.set_length()
        self.set_indices()
    
    def __getitem__(self, index):
        if index == 0 and self.shuffle:
            self.set_indices(self.shuffle)
        return self.get(self.indices[index])
    

def PipeLoader(dataset, batch_size = 0, repeat = 1, shuffle = False, prefetch = False, num_parallel_calls = True, dtype = None):
    """
    Convert tf pipeline (=torch dataloader)
    
    <example>
    > dataset = tfdet.dataset.Dataset(*args)
    > pipe = tfdet.dataset.PipeLoader(dataset)
    > pipe = tfdet.dataset.pipeline.args2dict(pipe) #for train_model
    > pipe = tfdet.dataset.pipeline.collect(pipe) #optional for semantic segmentation
    > pipe = tfdet.dataset.pipeline.cast(pipe)
    > pipe = tfdet.dataset.pipeline.key_map(pipe, batch_size = 16, shuffle = False, prefetch = True)
    > next(iter(dataset))
    """
    args = dataset[0]
    assert not isinstance(args, dict), "Dataset output is should not dictionary."
    
    if shuffle:
        dataset.shuffle = False
    
    indices = np.expand_dims(np.arange(len(dataset)), axis = -1)
    if dtype is None:
        dtype = tuple([tf.convert_to_tensor(v).dtype for v in ((args,) if not isinstance(args, tuple) else args)])
        if not isinstance(args, tuple):
            dtype = dtype[0]

    def load_iter_data(index = None):
        if index is None:
            index = [np.random.randint(len(dataset))]
        return dataset[index[0]]
    load_func = functools.partial(py_func, load_iter_data, Tout = dtype)
    return pipeline(indices, function = load_func,
                    batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                    num_parallel_calls = num_parallel_calls)

def GenPipeLoader(dataset, batch_size = 0, repeat = 1, shuffle = False, prefetch = False, num_parallel_calls = True, dtype = None):
    """
    Convert tf pipeline by generator (=torch dataloader) #so slow
    
    <example>
    > dataset = tfdet.dataset.Dataset(*args)
    > pipe = tfdet.dataset.GenPipeLoader(dataset)
    > pipe = tfdet.dataset.pipeline.args2dict(pipe) #for train_model
    > pipe = tfdet.dataset.pipeline.collect(pipe) #optional for semantic segmentation
    > pipe = tfdet.dataset.pipeline.cast(pipe)
    > pipe = tfdet.dataset.pipeline.key_map(pipe, batch_size = 16, shuffle = False, prefetch = True)
    > next(iter(dataset))
    """
    args = dataset[0]
    assert not isinstance(args, dict), "Dataset output is should not dictionary."
    
    if shuffle:
        dataset.shuffle = False
    
    if dtype is None:
        dtype = tuple([tf.convert_to_tensor(v).dtype for v in ((args,) if not isinstance(args, tuple) else args)])
        if not isinstance(args, tuple):
            dtype = dtype[0]
    
    def load_generator(dataset):
        return dataset
    pipe = tf.data.Dataset.from_generator(load_generator, dtype)
    return pipeline(pipe,
                    batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                    num_parallel_calls = num_parallel_calls)

class SequenceLoader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size = 0, num_parallel_calls = True):
        """
        Convert keras sequence (=torch dataloader)

        <example>
        > dataset = tfdet.dataset.Dataset(*args)
        > sequence = tfdet.dataset.SequenceLoader(dataset, batch_size = 16)
        > dataset[i] #or next(iter(dataset))
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_parallel_calls = max(num_parallel_calls if not isinstance(num_parallel_calls, bool) else (8 if num_parallel_calls else 0), 1)
        
        self.indices = [np.arange(i * max(self.batch_size, 1), min(len(self.dataset), (i + 1) * max(self.batch_size, 1))) for i in range(int(np.ceil(len(self.dataset) / max(self.batch_size, 1))))]

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        indices = self.indices[index]
        data = ThreadPool(min(self.num_parallel_calls, len(indices))).imap(lambda index: self.dataset[index], indices)
        if 0 < self.batch_size:
            data = self.dataset.stack(*data)
        else:
            data = list(data)[0]
        data = (data,) if not isinstance(data, tuple) else data
        data = [np.array(arg) if not isinstance(arg, np.ndarray) else arg for arg in data]
        return data[0] if len(data) == 1 else tuple(data)