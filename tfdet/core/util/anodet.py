import tensorflow as tf
import numpy as np

def feature_concat(x, y):
    b = tf.shape(x)[0]
    h1, w1, c1 = tf.keras.backend.int_shape(x)[1:]
    h2, w2, c2 = tf.keras.backend.int_shape(y)[1:]
    
    kernel_size = int(h1 / h2)
    x = tf.nn.space_to_depth(x, kernel_size)
    x = tf.reshape(x, [b, h2, w2, -1, c1])
    shape = [1 if index != 2 else tf.shape(x)[-2] for index in range(tf.keras.backend.ndim(x), 0, -1)]
    y = tf.tile(tf.expand_dims(y, axis = -2), shape)
    out = tf.concat([x, y], axis = -1)
    out = tf.reshape(out, [b, h2, w2, -1])
    out = tf.nn.depth_to_space(out, kernel_size)
    out = tf.reshape(out, [b, h1, w1, c1 + c2])
    return out
    
def feature_extract(feature, sampling_index = None, pool_size = 1, sub_sampling = False, concat = True, memory_reduce = True):
    if not isinstance(feature, list):
        feature = [feature]
    if isinstance(sampling_index, int):
        sampling_index = [sampling_index]
    feature = list(feature)
    
    if sub_sampling:
        feature.append(tf.nn.avg_pool(feature[-1], tf.keras.backend.int_shape(feature[-1])[-3], 1, padding = "VALID"))
    
    pad = 0
    for level in range(len(feature)):
        if 1 < pool_size:
            feature[level] = tf.nn.avg_pool(feature[level], pool_size, 1, padding = "SAME")
        if sampling_index is not None and (memory_reduce or not concat):
            h, w, c = tf.keras.backend.int_shape(feature[level])[1:]
            indices = tf.gather(sampling_index, tf.where(tf.logical_and(tf.greater_equal(sampling_index, pad), tf.less(sampling_index, c + pad))))[:, 0] - pad
            feature[level] = tf.reshape(tf.gather(feature[level], indices, axis = -1), [-1, h, w, len(indices)])
            pad += c
    
    if concat:
        fv = feature[0]
        for f in feature[1:]:
            fv = feature_concat(fv, f)
        if sampling_index is not None and not memory_reduce:
            h, w, c = tf.keras.backend.int_shape(fv)[1:]
            fv = tf.reshape(tf.gather(fv, sampling_index, axis = -1), [-1, h, w, len(sampling_index)])
        feature = [fv]
        
    if len(feature) == 1:
        feature = feature[0]
    return feature


def core_sampling(*args, n_sample = 3, n_feature = "auto", eps = 0.9, index = False):
    try:
        from sklearn.random_projection import SparseRandomProjection, johnson_lindenstrauss_min_dim
    except:
        print("If you want to use 'core_sampling', please install 'scikit-learn 0.13▲'")
        return
    if isinstance(n_sample, float):
        n_sample = int(len(args[0]) * n_sample)
    n_sample = max(min(n_sample, len(args[0])), 1 if len(args[0]) != 0 else 0)
    if n_feature == "auto":
        b, c = np.shape(args[0])
        n_feature = max(min(johnson_lindenstrauss_min_dim(b, eps = eps), c), 1)
    m = SparseRandomProjection(n_components = n_feature, eps = eps)
    trans_data = m.fit_transform(args[0])
    
    indices = []
    min_dist = None
    target = np.expand_dims(trans_data[0], axis = 0)
    for j in range(n_sample): #k center greedy
        dist = np.linalg.norm(trans_data - target, axis = -1, keepdims = True)
        min_dist = np.minimum(dist, min_dist) if min_dist is not None else dist
        min_index = np.argmax(min_dist)
        target = np.expand_dims(trans_data[min_index], axis = 0)
        min_dist[min_index] = 0
        indices.append(min_index)
    
    if not index:
        args = [np.array(arg)[indices] for arg in args]
        if len(args) == 1:
            args = args[0]
        return args
    else:
        return indices

def get_threshold(y_true, y_pred):
    try:
        from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
    except:
        print("If you want to use 'get_threshold', please install 'scikit-learn 0.14▲'")
        return
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1 = np.divide(2 * precision * recall, precision + recall, out = np.zeros_like(precision), where = (precision + recall) != 0)
    threshold = thresholds[np.argmax(f1)]
    return threshold