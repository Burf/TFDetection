import functools
import os

import cv2
import numpy as np
import tensorflow as tf

from .dataset import Dataset
from .util import load_json
from tfdet.core.util import pipeline, py_func

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

#from coco91 to coco80
CATEGORY_ID = [0, 
               1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

memory = {}

def clear(path = None):
    global memory
    if isinstance(path, str):
        key = os.path.basename(path)
        if key in memory:
            del memory[key]
    else:
        memory.clear()
    return memory

def get(path, refresh = False):
    try:
        from pycocotools.coco import COCO
    except Exception as e:
        print("If you want to use 'coco dataset', please install 'pycocotools'")
        raise e
    
    global memory
    key = os.path.abspath(path)
    if refresh or key not in memory:
        coco = COCO(path)
        memory[key] = coco
    else:
        coco = memory[key]
    return coco

def merge_multi_segment(segments):
    """
    https://github.com/ultralytics/JSON2YOLO/blob/master/general_json2yolo.py
    """
    def min_index(arr1, arr2):
        dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
        return np.unravel_index(np.argmin(dis, axis=None), dis.shape)
    
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s

def convert_contour(segments):
    if 1 < len(segments):
        s = merge_multi_segment(segments)
        #s = np.concatenate(s, axis = 0)
        s = np.vstack(s).astype(np.float32)
    elif len(segments) == 1:
        s = [j for i in segments for j in i]  # all segments concatenated
        s = np.reshape(np.array(s, dtype = np.float32), [-1, 2])
    else:
        s = np.zeros([0, 2], dtype = np.float32)
    return s

def contour2instance(contour, image_shape, ignore_label = 0):
    mask_true = np.full([len(contour), *image_shape, 1], ignore_label, dtype = np.uint8)
    for i, cont in enumerate(contour):
        cv2.drawContours(mask_true[i], np.round(np.reshape(cont, [1, -1, 2])).astype(np.int32), -1, 1, -1)
    return mask_true

def load_info(coco, data_path, x_true, mask = False, crowd = False, label = LABEL, cat_ids = None, cat2label = None):       
    if isinstance(coco, str):
        coco = get(coco)
    if cat_ids is None:
        cat_ids = coco.getCatIds(label)
    if cat2label is None:
        cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        
    info = coco.loadImgs([x_true])[0]
    filename = info["file_name"]
    height, width = info["height"], info["width"]
    anno_id = coco.getAnnIds([x_true], iscrowd = crowd)
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
        y_true.append([label[int(cat2label[a["category_id"]] + 1)] if cat2label is not None else int(a["category_id"])])
        bbox_true.append([int(round(x1)), int(round(y1)), int(round(x1 + w)), int(round(y1 + h))])    
        if mask:
            #mask_true.append(a["segmentation"])
            mask_true.append(convert_contour(a["segmentation"]))
    y_true = np.array(y_true, dtype = np.object0) if 0 < len(y_true) else np.zeros((0, 1), dtype = np.object0)
    bbox_true = np.array(bbox_true, dtype = np.int32) if 0 < len(bbox_true) else np.zeros((0, 4), dtype = np.int32)
    return (x_true, y_true, bbox_true) if not mask else (x_true, y_true, bbox_true, mask_true)

def load_instance(mask_true, image_shape):
    if False:
        try:
            from pycocotools import mask as maskUtils
        except Exception as e:
            print("If you want to use 'coco dataset', please install 'pycocotools'")
            raise e
        
        mask = []
        for seg in mask_true:
            if isinstance(seg, list):
                rles = maskUtils.frPyObjects(seg, *image_shape[:2])
                rle = maskUtils.merge(rles)
            else:
                if isinstance(seg["counts"], list):
                    rle = maskUtils.frPyObjects([seg], *image_shape[:2])
                else:
                    rle = [rle]
            mask.append(maskUtils.decode(rle))
        mask_true = np.expand_dims(np.stack(mask, axis = 0).astype(np.uint8), axis = -1) if 0 < len(mask) else np.zeros((0, *image_shape[:2], 1), dtype = np.uint8)
    else:
        mask_true = contour2instance(mask_true, image_shape)
    return mask_true

def load_annotation(x_true, y_true, bbox_true, mask_true = None):
    if mask_true is not None:
        mask_true = load_instance(mask_true, np.shape(cv2.imread(x_true, -1))[:2])
    result = (x_true, y_true, bbox_true)
    if mask_true is not None:
        result = (x_true, y_true, bbox_true, mask_true)
    return result

def load_dataset(path, data_path, mask = False, crowd = False, label = LABEL,
                 transform = None, refresh = False, shuffle = False,
                 cache = None):
    """
    https://cocodataset.org
    
    <example>
    1. all-in-one
    > dataset = tfdet.dataset.coco.load_dataset("./coco/annotations/instances_train2017.json", "./coco/train2017",
                                                transform = [load, resize, pad,
                                                             filter_annotation, label_encode, normalize]
                                                mask = False, crowd = False,
                                                shuffle = False, cache = "coco_train.cache")
    > dataset[i] #or next(iter(dataset))
    
    2. split
    > dataset = tfdet.dataset.coco.load_dataset("./coco/annotations/instances_train2017.json", "./coco/train2017",
                                                mask = False, crowd = False,
                                                shuffle = False, cache = "coco_train.cache")
    > dataset = tfdet.dataset.Dataset(dataset,
                                      transform = [load, resize, pad,
                                                   filter_annotation, label_encode, normalize])
    > dataset[i] #or next(iter(dataset))
        
    3. dataset to pipe
    > pipe = tfdet.dataset.PipeLoader(dataset)
    > pipe = tfdet.dataset.pipeline.args2dict(pipe) #optional for object detection
    > pipe = tfdet.dataset.pipeline.collect(pipe) #optional for semantic segmentation
    > pipe = tfdet.dataset.pipeline.cast(pipe)
    > pipe = tfdet.dataset.pipeline.key_map(pipe, batch_size = 16, shuffle = False, prefetch = True)
    > next(iter(dataset))
    """
    transform = ([transform] if not isinstance(transform, (list, tuple)) else list(transform)) if transform is not None else []
    
    if isinstance(cache, str) and os.path.exists(cache):
        return Dataset(transform = [load_annotation] + transform, shuffle = shuffle, cache = cache, keys = ["x_true", "y_true", "bbox_true", "mask_true"])
    else:
        coco = get(path, refresh = refresh)
        cat_ids = coco.getCatIds(label)
        cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        ids = np.array(coco.getImgIds())

        preprocess_func = functools.partial(load_info, path, data_path, mask = mask, crowd = crowd, label = label, cat_ids = cat_ids, cat2label = cat2label)
        return Dataset(ids, preprocess = preprocess_func, transform = [load_annotation] + transform, shuffle = shuffle, cache = cache, keys = ["x_true", "y_true", "bbox_true", "mask_true"])

def convert_tfds_to_tfdet(data, crowd = False, label = LABEL[1:]):
    x_true = data["image"]
    obj = data["objects"]
    y_true = tf.expand_dims(obj["label"], axis = -1)
    if label is not None:
        y_true = tf.gather(label, y_true)
    bbox_true = obj["bbox"]
    area = obj["area"]
    if crowd is not None:
        if crowd:
            indices = tf.where(obj["is_crowd"])[:, 0]
        else:
            indices = tf.where(tf.logical_not(obj["is_crowd"]))[:, 0]
        y_true = tf.gather(y_true, indices)
        bbox_true = tf.gather(bbox_true, indices)
        area = tf.gather(area, indices)
    indices = tf.where(0 < area)[:, 0]
    y_true = tf.gather(y_true, indices)
    bbox_true = tf.gather(bbox_true, indices)
    return x_true, y_true, bbox_true

def tfds_to_tfdet(tfds_pipe, crowd = False, label = LABEL[1:],
                  batch_size = 0, repeat = 1, shuffle = False, prefetch = False,
                  cache = False, num_parallel_calls = True):
    """
    <example>
    import tfdet
    import tensorflow_datasets as tfds
    pipe = tfds.load("coco/2017", split = tfds.Split.VALIDATION)
    pipe = tfdet.dataset.coco.tfds_to_tfdet(pipe, crowd = False)

    x_true, y_true, bbox_true = next(iter(pipe)) #x_true:(640, 480, 3), y_true:(4, 1), bbox_true:(4, 4)
    """
    func = functools.partial(convert_tfds_to_tfdet, crowd = crowd, label = label)
    pipe = tfds_pipe.map(func)
    return pipeline(pipe,
                    batch_size = batch_size, repeat = repeat, shuffle = shuffle, prefetch = prefetch,
                    cache = cache, num_parallel_calls = num_parallel_calls)

def convert_format(path, y_true, bbox_true, coco = True):
    """
    path = image path
    y_true = logits (N, n_class) (index-0 is background)
    bbox_true = (N, 4)
    """
    if np.ndim(y_true) == 3:
        y_true = y_true[0]
        bbox_true = bbox_true[0]
    bbox_true = np.array(bbox_true) if not isinstance(bbox_true, np.ndarray) else bbox_true
    valid_indices = np.where(0 < np.max(bbox_true, axis = -1))
    y_true = (np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true)[valid_indices]
    bbox_true = bbox_true[valid_indices]
    if np.shape(y_true)[-1] == 1:
        label_true = y_true
        score_true = np.ones_like(label_true)
    else:
        label_true = np.expand_dims(np.argmax(y_true, axis = -1), axis = -1)
        score_true = np.max(y_true, axis = -1, keepdims = True)
    if True: #remove background
        flag = label_true[..., 0] != ignore_label
        label_true = label_true[flag]
        score_true = score_true[flag]
        bbox_true = bbox_true[flag]
    if 0 < len(bbox_true) and np.max(bbox_true) < 2:
        h, w, c = np.shape(cv2.imread(path))
        bbox_true = np.multiply(bbox_true, [w, h, w, h], dtype = np.float32)
    name = os.path.splitext(os.path.basename(path))[0]
    image_id = int(name) if name.isnumeric() else name
    x1, y1, x2, y2 = np.split(bbox_true, 4, axis = -1)
    w, h = x2 - x1, y2 - y1
    bbox_true = np.hstack([x1, y1, w, h])
    data = [{"image_id": image_id,
             "category_id": CATEGORY_ID[int(label_true[index])] if coco else int(label_true[index]),
             "score": np.round(float(score_true[index][0]), 5),
             "bbox": [np.round(float(p), 3) for p in bbox_true[index]]}
               for index in range(len(label_true))]
    return data

def coco_evaluate(anno_path, pred_path, coco = True, mode = "bbox"):
    """
    #https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    
    anno_path = "./coco/annotations/instances_val2017.json"
    pred_path = list (N,) or pred_json_path
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as e:
        print("If you want to use 'load_coco', please install 'pycocotools'")
        raise e
        
    pred_anno = load_json(pred_path) if isinstance(pred_path, str) else pred_path
    if np.ndim(pred_anno[0]) == 1:
        pred_anno = np.concatenate(pred_anno, axis = 0)
    if isinstance(pred_anno, np.ndarray):
        pred_anno = [p for p in pred_anno]
    
    coco_anno = COCO(anno_path)
    coco_pred = coco_anno.loadRes(pred_anno)
    coco_eval = COCOeval(coco_anno, coco_pred, mode)
    if coco:
        unique_ids, indices = np.unique([p["image_id"] for p in pred_anno], return_index = True)
        image_ids = unique_ids[np.argsort(indices)].tolist()
        coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mean_average_precision, mean_average_precision_50 = coco_eval.stats[:2]
    return mean_average_precision, mean_average_precision_50