import tensorflow as tf

def key_map(x_true, y_true = None, bbox_true = None, mask_true = None, map = {"x_true":"x_true", "y_true":"y_true", "bbox_true":"bbox_true", "mask_true":"mask_true"}):
    if isinstance(x_true, dict):
        x_true = {map[k] if k in map else k:v for k, v in x_true.items()}
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def collect(x_true, y_true = None, bbox_true = None, mask_true = None, keys = ["x_true", "y_true", "bbox_true", "mask_true"]):
    if isinstance(x_true, dict):
        x_true = {k:x_true[k] for k in keys if k in x_true}
    else:
        if "x_true" not in keys:
            x_true = None
        if "y_true" not in keys:
            y_true = None
        if "bbox_true" not in keys:
            bbox_true = None
        if "mask_true" not in keys:
            mask_true = None
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def cast(x_true, y_true = None, bbox_true = None, mask_true = None, map = {"x_true":tf.float32, "y_true":tf.float32, "bbox_true":tf.float32, "mask_true":tf.float32}):
    if isinstance(x_true, dict):
        x_true = {k:tf.cast(v, map[k]) if k in map and map[k] is not None else v for k, v in x_true.items()}
    else:
        if "x_true" in map and map["x_true"] is not None:
            x_true = tf.cast(x_true, map["x_true"])
        if y_true is not None and "y_true" in map and map["y_true"] is not None:
            y_true = tf.cast(y_true, map["y_true"])
        if bbox_true is not None and "bbox_true" in map and map["bbox_true"] is not None:
            bbox_true = tf.cast(bbox_true, map["bbox_true"])
        if mask_true is not None and "mask_true" in map and map["mask_true"] is not None:
            mask_true = tf.cast(mask_true, map["mask_true"])
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def reshape(x_true, y_true = None, bbox_true = None, mask_true = None, map = {"x_true":None, "y_true":None, "bbox_true":None, "mask_true":None}):
    if isinstance(x_true, dict):
        x_true = {k:tf.reshape(v, map[k]) if k in map and map[k] is not None else v for k, v in x_true.items()}
    else:
        if "x_true" in map and map["x_true"] is not None:
            x_true = tf.reshape(x_true, map["x_true"])
        if y_true is not None and "y_true" in map and map["y_true"] is not None:
            y_true = tf.reshape(y_true, map["y_true"])
        if bbox_true is not None and "bbox_true" in map and map["bbox_true"] is not None:
            bbox_true = tf.reshape(bbox_true, map["bbox_true"])
        if mask_true is not None and "mask_true" in map and map["mask_true"] is not None:
            mask_true = tf.reshape(mask_true, map["mask_true"])
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def args2dict(x_true, y_true = None, bbox_true = None, mask_true = None, keys = ["x_true", "y_true", "bbox_true", "mask_true"]):
    if not isinstance(x_true, dict):
        x_true = {k:v for k, v in zip(keys, [x_true, y_true, bbox_true, mask_true]) if v is not None}
        y_true = bbox_true = mask_true = None
    result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result

def dict2args(x_true, y_true = None, bbox_true = None, mask_true = None, keys = None):
    if isinstance(x_true, dict):
        if keys is not None:
            x_true = {k:x_true[k] for k in keys if k in x_true}
        result = list(x_true.values())
    else:
        result = [v for v in [x_true, y_true, bbox_true, mask_true] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result