import functools
import os

import numpy as np
import tensorflow as tf

from tfdet.core.util import pipeline

LABEL = ["background", #background
         "person", "bicycle", "car", "motorcycle", "airplane", "bus",
         "train", "truck", "boat", "traffic light", "fire hydrant",
         "stop sign", "parking meter", "bench", "bird", "cat", "dog",
         "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
         "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
         "skis", "snowboard", "sports ball", "kite", "baseball bat",
         "baseball glove", "skateboard", "surfboard", "tennis racket",
         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
         "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
         "hot dog", "pizza", "donut", "cake", "chair", "couch",
         "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
         "mouse", "remote", "keyboard", "cell phone", "microwave",
         "oven", "toaster", "sink", "refrigerator", "book", "clock",
         "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

COLOR = [(0, 0, 0),
         (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
         (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
         (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
         (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
         (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
         (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
         (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
         (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
         (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
         (134, 134, 103), (145, 148, 174), (255, 208, 186),
         (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
         (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
         (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
         (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
         (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
         (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
         (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
         (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
         (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
         (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
         (191, 162, 208)]

def load_data(path, data_path, mask = False, crowd = False, label = LABEL, shuffle = False):
    """
    https://cocodataset.org
    
    <example>
    path = "./coco/annotations/instances_train2017.json"
    data_path = "./coco/train2017"
    mask = with instance mask_true
    crowd = iscrowd
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools import mask as maskUtils
    except Exception as e:
        print("If you want to use 'load_coco', please install 'pycocotools'")
        raise e

    coco = COCO(path)
    cat_ids = coco.getCatIds(label)
    cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    ids = coco.getImgIds()
    if shuffle:
        np.random.shuffle(ids)
    for id in ids:
        info = coco.loadImgs([id])[0]
        filename = info["file_name"]
        height, width = info["height"], info["width"]
        anno_id = coco.getAnnIds([id], iscrowd = crowd)
        anno = coco.loadAnns(anno_id)
        x_true = os.path.join(data_path, filename)
        y_true = []
        bbox_true = []
        mask_true = []
        for a in anno:
            x1, y1, w, h = a["bbox"]
            inter_w = max(0, min(x1 + w, width) - max(x1, 0))
            inter_h = max(0, min(y1 + h, height) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if a["area"] <= 0 or w < 1 or h < 1:
                continue
            if a["category_id"] not in cat_ids:
                continue
                
            y_true.append([label[int(cat2label[a["category_id"]] + 1)]])
            bbox_true.append([int(round(x1)), int(round(y1)), int(round(x1 + w)), int(round(y1 + h))])
            if mask:
                seg = a["segmentation"]
                if isinstance(seg, list):
                    rles = maskUtils.frPyObjects(seg, height, width)
                    rle = maskUtils.merge(rles)
                else:
                    if isinstance(seg["counts"], list):
                        rle = maskUtils.frPyObjects([seg], height, width)
                    else:
                        rle = [rle]
                mask_true.append(maskUtils.decode(rle))
        y_true = np.array(y_true) if 0 < len(y_true) else np.zeros((0, 1), dtype = str)
        bbox_true = np.array(bbox_true) if 0 < len(bbox_true) else np.zeros((0, 4), dtype = int)
        result = (x_true, y_true, bbox_true)
        if mask:
            mask_true = np.expand_dims(mask_true, axis = -1).astype(np.float32) if 0 < len(bbox_true) else np.zeros((0, h, w, 1), dtype = np.float32)
            result = (x_true, y_true, bbox_true, mask_true)
        yield result
        
def load_pipe(path, data_path, mask = False, crowd = False, label = LABEL, shuffle = False,
              batch_size = 0, repeat = 1, prefetch = False,
              cache = None, num_parallel_calls = None):
    """
    https://cocodataset.org
    
    <example>
    path = "./coco/annotations/instances_train2017.json"
    data_path = "./coco/train2017"
    mask = with instance mask_true
    crowd = iscrowd
    """
    generator = functools.partial(load_data, path, data_path, mask = mask, crowd = crowd, label = label, shuffle = shuffle)
    dtype = (tf.string, tf.string, tf.int32, tf.float32) if mask else (tf.string, tf.string, tf.int32)
    pipe = tf.data.Dataset.from_generator(generator, dtype)
    return pipeline(pipe, batch_size = batch_size, repeat = repeat, shuffle = False, prefetch = prefetch,
                    cache = cache, num_parallel_calls = num_parallel_calls)