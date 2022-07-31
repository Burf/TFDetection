import cv2
import numpy as np

from .xml import xml2dict

def load_pascal_voc(path, bbox = None):
    label = path
    if isinstance(path, str):
        anno = xml2dict(path)["annotation"]
        if "object" in anno:
            objs = anno["object"]
            label = []
            bbox = []
            for obj in objs if isinstance(objs, list) else [objs]:
                label.append([obj["name"]])
                bbox.append([int(round(float(obj["bndbox"][k]))) for k in ["xmin", "ymin", "xmax", "ymax"]])
            label = np.array(label, dtype = str)
            bbox = np.array(bbox, dtype = int)
        else:
            label = np.zeros([0, 1], dtype = str)
            bbox = np.zeros([0, 4], dtype = int)
    result = [v for v in [label, bbox] if v is not None]
    result = result[0] if len(result) == 1 else tuple(result)
    return result
    
def pascal_voc(path, y_true, bbox_true):
    y_true = np.squeeze(y_true)
    bbox_true = np.squeeze(bbox_true)
    if np.ndim(y_true) == 0:
        y_true = [y_true]
        bbox_true = [bbox_true]
    h, w, c = np.shape(cv2.imread(path))
    data = {"annotation": {"folder": os.path.basename(os.path.dirname(path)),
                           "filename": os.path.basename(path),
                           "path": path,
                           "source": {"database": "Unknown"},
                           "size": {"width": str(w), "height": str(h), "depth": str(c)},
                           "segmented": "0",
                           "object":obj}}
    try:
        if np.max(bbox_true) <= 1:
            bbox_true = np.multiply(bbox_true, [w, h, w, h]).astype(np.int32)
        obj = [{"name": str(y_true[index]),
                "pose": "Unspecified",
                "truncated": "0",
                "difficult": "0",
                "bndbox": {"xmin": str(bbox_true[index][0]), "ymin": str(bbox_true[index][1]), "xmax": str(bbox_true[index][2]), "ymax": str(bbox_true[index][3])}}
               for index in range(len(y_true))]
        if len(obj) == 1:
            obj = obj[0]
        data["annotation"]["object"] = obj
    except:
        pass
    return data