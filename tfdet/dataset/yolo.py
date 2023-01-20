import functools

import cv2
import numpy as np

from tfdet.core.util import dict_function
from tfdet.dataset.transform import load, resize, pad, filter_annotation, mosaic, mosaic9, mix_up, copy_paste, yolo_hsv, random_perspective, random_flip, compose
from .dataset import Dataset

class YoloDataset(Dataset):
    def __init__(self, *args, transform = None, preprocess = None, 
                 image_shape = [640, 640], keep_ratio = True, pad_val = 114,
                 perspective = 0., rotate = 0., translate = 0.2, scale = 0.9, shear = 0.,
                 h = 0.015, s = 0.7, v = 0.4,
                 max_paste_count = 30, scale_range = [0.0625, 0.75], clip_object = True, replace = True, random_count = False, label = None,
                 min_scale = 2, min_instance_area = 1, iou_threshold = 0.3, copy_min_scale = 2, copy_min_instance_area = 1, copy_iou_threshold = 0.3, p_copy_paste_flip = 0.5, method = cv2.INTER_LINEAR,
                 p_mosaic = 1., p_mix_up = 0.15, p_copy_paste = 0., p_flip = 0.5, p_mosaic9 = 0.2,
                 min_area = 0., min_visibility = 0., e = 1e-12,
                 shuffle = False, cache = None):
        """
        args > x_true, y_true, bbox_true, mask_true(optional) style args or dataset
        
        <example>
        1. basic
        > dataset = tfdet.dataset.YoloDataset(x_true, y_true, bbox_true, mask_true(optional),
                                              **kwargs,
                                              transform = [filter_annotation, label_encode, normalize], #post-apply transform
                                              preprocess = [], #pre-apply transform
                                              shuffle = False, #when item 0 is called, shuffle indices.(Recommended by 1 GPU)
                                              cache = "dataset.cache", #save cache after preprocess)
        > dataset[i] #or next(iter(dataset))
        
        2. dataset
        > dataset = tfdet.dataset.coco.load_dataset("./coco/annotations/instances_train2017.json", "./coco/train2017",
                                                    mask = False, crowd = False,
                                                    shuffle = False, cache = "coco_train.cache")
        > dataset = tfdet.dataset.YoloDataset(dataset,
                                              **kwargs,
                                              transform = [filter_annotation, label_encode, normalize])
        > dataset[i] #or next(iter(dataset))
        
        3. dataset to pipe
        > pipe = tfdet.dataset.PipeLoader(dataset)
        > pipe = tfdet.dataset.pipeline.args2dict(pipe) #optional for object detection
        > pipe = tfdet.dataset.pipeline.collect(pipe) #optional for semantic segmentation
        > pipe = tfdet.dataset.pipeline.cast(pipe)
        > pipe = tfdet.dataset.pipeline.key_map(pipe, batch_size = 16, shuffle = False, prefetch = True)
        > next(iter(dataset))
        """
        super(YoloDataset, self).__init__(*args, preprocess = preprocess, shuffle = shuffle, cache = cache, keys = ["x_true", "y_true", "bbox_true", "mask_true"])
        self.old_get = super(YoloDataset, self).get
        self.postprocess = [transform] if not isinstance(transform, (list, tuple)) else transform
        
        self.image_shape, self.keep_ratio, self.pad_val = image_shape, keep_ratio, pad_val
        self.perspective, self.rotate, self.translate, self.scale, self.shear = perspective, rotate, translate, scale, shear
        self.h, self.s, self.v = h, s, v
        self.max_paste_count, self.scale_range, self.clip_object, self.replace, self.random_count, self.label = max_paste_count, scale_range, clip_object, replace, random_count, label
        self.min_scale, self.min_instance_area, self.iou_threshold, self.copy_min_scale, self.copy_min_instance_area, self.copy_iou_threshold, self.p_copy_paste_flip, self.method = min_scale, min_instance_area, iou_threshold, copy_min_scale, copy_min_instance_area, copy_iou_threshold, p_copy_paste_flip, method
        self.p_mosaic, self.p_mix_up, self.p_copy_paste, self.p_flip, self.p_mosaic9 = p_mosaic, p_mix_up, p_copy_paste, p_flip, p_mosaic9
        self.min_area, self.min_visibility, self.e = min_area, min_visibility, e
        
    def load_image(self, index):
        if isinstance(self.args[0], Dataset):
            args = self.args[0].get(index)
        else:
            args = self.old_get(index)
        args = (args,) if not isinstance(args, tuple) else args
        return dict_function(self.keys)([load, 
                                         functools.partial(resize, image_shape = self.image_shape, keep_ratio = self.keep_ratio, method = self.method)])(*args)
        
    def get(self, index, transform = None, preprocess = True):
        if transform is not None and preprocess:
            if isinstance(self.args[0], Dataset):
                args = self.args[0].get(index)
            else:
                args = self.old_get(index, transform = transform)
        else:
            random_perspective_func = functools.partial(random_perspective, image_shape = self.image_shape, perspective = self.perspective, rotate = self.rotate, translate = self.translate, scale = self.scale, shear = self.shear, pad_val = self.pad_val, min_area = self.min_area, min_visibility = self.min_visibility, e = self.e)
            filter_annotation_func = functools.partial(filter_annotation, min_scale = self.min_scale, min_instance_area = self.min_instance_area)
            if np.random.random() < self.p_mosaic:
                mosaic_func = functools.partial(mosaic, image_shape = np.multiply(self.image_shape, 2).astype(int), pad_val = self.pad_val, min_area = self.min_area, min_visibility = self.min_visibility, e = self.e)
                mosaic9_func = functools.partial(mosaic9, image_shape = np.multiply(self.image_shape, 2).astype(int), pad_val = self.pad_val, min_area = self.min_area, min_visibility = self.min_visibility, e = self.e)
                
                if np.random.random() < (1 - self.p_mosaic9):
                    base_indices = [index] + np.random.choice(self.indices, 3, replace = True).tolist()
                    base_mosaic_func = mosaic_func
                else:
                    base_indices = [index] + np.random.choice(self.indices, 8, replace = True).tolist()
                    base_mosaic_func = mosaic9_func
                    
                mix, mix_indices = False, []
                if np.random.random() < self.p_mix_up:
                    mix = True
                    if np.random.random() < (1 - self.p_mosaic9):
                        mix_indices = np.random.choice(self.indices, 4, replace = True).tolist()
                        mix_mosaic_func = mosaic_func
                    else:
                        mix_indices = np.random.choice(self.indices, 9, replace = True).tolist()
                        mix_mosaic_func = mosaic9_func
                
                args = [(index, self.load_image(index)) for index in np.unique(base_indices + mix_indices)]
                unique_indices = [arg[0] for arg in args]
                args = [(arg[1],) if not isinstance(arg[1], tuple) else arg[1] for arg in args]
                store = {k:v for k, v in zip(unique_indices, args)}
                
                args = self.stack(*[store[i] for i in base_indices])
                args = dict_function(self.keys)([base_mosaic_func, 
                                                 random_perspective_func, 
                                                 #filter_annotation_func,
                                                ])(*args)
                args = (args,) if not isinstance(args, tuple) else args
                
                if mix:
                    sample_args = self.stack(*[store[i] for i in mix_indices])
                    sample_args = dict_function(self.keys)([mix_mosaic_func, 
                                                            random_perspective_func, 
                                                            #filter_annotation_func,
                                                           ])(*sample_args)
                    sample_args = (sample_args,) if not isinstance(sample_args, tuple) else sample_args
                    args = self.stack(args, sample_args)
                    #args = mix_up(*args)
                    args = dict_function(self.keys)(mix_up)(*args)
                    args = (args,) if not isinstance(args, tuple) else args
                del store
            else:
                args = self.load_image(index)
                args = (args,) if not isinstance(args, tuple) else args
                args = dict_function(self.keys)([functools.partial(pad, image_shape = self.image_shape, max_pad_size = 0, pad_val = self.pad_val),
                                                 random_perspective_func,
                                                 #filter_annotation_func,
                                                ])(*args)
                args = (args,) if not isinstance(args, tuple) else args
                
            args = dict_function(self.keys)([filter_annotation_func,
                                             functools.partial(yolo_hsv, h = self.h, s = self.s, v = self.v)])(*args)
            args = (args,) if not isinstance(args, tuple) else args
            
            if np.random.random() < self.p_copy_paste and ((isinstance(args[0], dict) and 1 < len(args[0])) or not (isinstance(args[0], dict) and 1 < len(args))):
                cp_indices = np.random.choice(self.indices, 4, replace = True)
                sample_args = [(index, self.load_image(index)) for index in np.unique(cp_indices)]
                unique_indices = [arg[0] for arg in sample_args]
                sample_args = [(arg[1],) if not isinstance(arg[1], tuple) else arg[1] for arg in sample_args]
                store = {k:v for k, v in zip(unique_indices, sample_args)}

                sample_args = [store[i] for i in cp_indices]
                for _ in range(10):
                    object_count = 0
                    if isinstance(args[0], dict):
                        for s in sample_args:
                            if "bbox_true" in s[0]:
                                object_count += len(s[0]["bbox_true"])
                            elif "mask_true" in s[0] and 3 < np.ndim(s[0]["mask_true"]):
                                object_count += len(s[0]["mask_true"])
                            elif "y_true" in s[0] and np.ndim(s[0]["y_true"]) == 2:
                                object_count += len(s[0]["y_true"])
                    else:
                        for s in sample_args:
                            for ss in s[1:]:
                                if np.ndim(ss) in [2, 4]:
                                    object_count += len(ss)
                                    break
                    if self.max_paste_count <= object_count:
                        break
                    index = np.random.choice(self.indices, 1, replace = True)[0]
                    sample_args2 = store[index] if index in store else self.load_image(index)
                    sample_args2 = (sample_args2,) if not isinstance(sample_args2, tuple) else sample_args2
                    sample_args.append(sample_args2)
                    if index not in store:
                        store[index] = sample_args2
                object_count = 0
                if isinstance(args[0], dict):
                    for s in sample_args:
                        if "bbox_true" in s[0]:
                            object_count += len(s[0]["bbox_true"])
                        elif "mask_true" in s[0] and 3 < np.ndim(s[0]["mask_true"]):
                            object_count += len(s[0]["mask_true"])
                        elif "y_true" in s[0] and np.ndim(s[0]["y_true"]) == 2:
                            object_count += len(s[0]["y_true"])
                else:
                    for s in sample_args:
                        for ss in s[1:]:
                            if np.ndim(ss) in [2, 4]:
                                object_count += len(ss)
                                break
                if 0 < object_count:
                    args = self.stack(args, *sample_args)
                    args = dict_function(self.keys)(copy_paste)(*args, 
                                                                max_paste_count = self.max_paste_count, scale_range = self.scale_range, clip_object = self.clip_object, replace = self.replace, random_count = self.random_count, label = self.label, min_scale = self.min_scale, min_instance_area = self.min_instance_area, iou_threshold = self.iou_threshold, copy_min_scale = self.copy_min_scale, copy_min_instance_area = self.copy_min_instance_area, copy_iou_threshold = self.copy_iou_threshold, p_flip = self.p_copy_paste_flip, method = self.method, 
                                                                min_area = self.min_area, min_visibility = self.min_visibility, e = self.e)
                    args = (args,) if not isinstance(args, tuple) else args
                del store, sample_args
            args = dict_function(self.keys)(random_flip)(*args, p = self.p_flip, mode = "horizontal")
        args = (args,) if not isinstance(args, tuple) else args
        
        postprocess = self.postprocess if transform is None else transform
        for j in range(len(postprocess)):
            func = postprocess[j]
            if callable(func):
                if hasattr(func, "func") and hasattr(func.func, "sample_size"):
                    sample_size = func.func.sample_size
                elif hasattr(func, "sample_size"):
                    sample_size = func.sample_size
                else:
                    sample_size = None
                if sample_size is not None:
                    sample_args= []
                    if 0 < sample_size:
                        sample_indices = np.random.choice(self.indices, sample_size, replace = True)
                        sample_args = [self.get(index, transform = postprocess[:j], preprocess = False) for index in sample_indices]
                    args = self.stack(args, *sample_args)
                args = dict_function(self.keys)(func)(*args)
                if not isinstance(args, tuple):
                    args = (args,)
        #args = [np.ascontiguousarray(arg) if isinstance(arg, np.ndarray) else arg for arg in args]
        return args[0] if len(args) == 1 else tuple(args)