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