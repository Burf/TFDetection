import tensorflow as tf
import numpy as np

from ..bbox import overlap_bbox_numpy as overlap_bbox

def generate_hist_scale(bbox_true, count = 5, decimal = 4):
    bbox_true = tf.reshape(bbox_true, (-1, 4))
    indices = tf.where(tf.reduce_max(tf.cast(0 < bbox_true, tf.int32), axis = -1))[:, 0]
    bbox_true = tf.gather(bbox_true, indices)

    h = bbox_true[:, 3] - bbox_true[:, 1]
    w = bbox_true[:, 2] - bbox_true[:, 0]
    h, w = np.histogram2d(h, w, bins = count)[1:]
    scale = np.stack([h, w], axis = -1)
    scale = np.mean([scale[:count], scale[-count:]], axis = 0)
    return np.sort(np.round(scale, decimal), axis = 0)

def generate_uniform_scale(min = 0.03125, max = 0.5, count = 5):
    return [min + (max - min) / (count - 1) * index for index in range(count)]

def generate_kmeans_scale(bbox_true, k = 5, decimal = 4, method = np.median, missing_value = 0., mode = "normal"):
    bbox_true = np.reshape(bbox_true, [-1, 4]).astype(np.float32)
    bbox_true = bbox_true[np.max(0 < bbox_true, axis = -1)]
    wh = bbox_true - np.tile(bbox_true[..., :2], 2) #x1, y1, x2, y2 -> 0, 0, w, h
    
    n_bbox = len(wh)
    last_nearest = np.zeros((n_bbox,))

    k = min(n_bbox, k)
    cluster = wh[np.random.choice(n_bbox, k, replace = False)]
    while True:
        overlaps = np.transpose(overlap_bbox(cluster, wh, mode = mode)) #(n_bbox, k)
        cur_nearest = np.argmin(1 - overlaps, axis = 1)
        if np.all(last_nearest == cur_nearest):
            break
        for index in range(k):
            target_wh = wh[cur_nearest == index]
            cluster[index] = method(target_wh, axis = 0) if 0 < len(target_wh) else missing_value
        last_nearest = cur_nearest
    return np.sort(np.round(cluster[..., 2:], decimal), axis = 0)