import tensorflow as tf
import numpy as np

def generate_anchors(feature, image_shape = [1024, 1024], scale = [0.03125, 0.0625, 0.125, 0.25, 0.5], ratio = [0.5, 1, 2], normalize = True, auto_scale = True, flatten = True, concat = True):
    """
    feature = feature or [features] or shape or [shapes]
    scale = anchor_size scale
    ratio = anchor_height * np.sqrt(ratio), anchor_weight / np.sqrt(ratio)
    """
    if tf.is_tensor(feature) or not isinstance(feature, list) or isinstance(feature[0], int):
        feature = [feature]
    if not isinstance(scale, list):
        scale = [scale]
    if not isinstance(ratio, list):
        ratio = [ratio]

    if not isinstance(scale[0], list):
        if auto_scale and (len(scale) % len(feature)) == 0:
            scale = tf.split(scale, len(feature))
        else:
            scale = [scale] * len(feature)
    if not isinstance(ratio[0], list):
        ratio = [ratio] * len(feature)

    out = []
    for x, scale, ratio in zip(feature, scale, ratio):
        if not isinstance(scale, list) and not (tf.is_tensor(scale) and tf.keras.backend.ndim(scale) != 0):
            scale = [scale]
        scale = [scale, scale]
        
        feature_shape = x
        if tf.is_tensor(x) and 2 < tf.keras.backend.ndim(x) or (not tf.is_tensor(x) and 2 < np.ndim(x)):
            feature_shape = tf.shape(x) if tf.keras.backend.int_shape(x)[-3] is None else tf.keras.backend.int_shape(x)
        if 2 < np.shape(feature_shape)[0]:
            feature_shape = feature_shape[-3:-1]
        
        stride = [1, 1]
        if 1 < tf.reduce_max(scale[0]):
            if normalize:
                shape = image_shape
                ndim = (tf.keras.backend.ndim(shape) if tf.is_tensor(shape) else np.ndim(shape)) - np.ndim(scale)
                for _ in range(ndim):
                    shape = tf.expand_dims(shape, axis = -1)
                scale = tf.divide(tf.cast(scale, tf.float32), tf.cast(shape, tf.float32))
            else:
                stride = image_shape
        
        # Generate base anchor
        scale = tf.cast(tf.expand_dims(scale, axis = 1), tf.float32)
        ratio = tf.cast(tf.expand_dims(ratio, axis = 1), tf.float32)

        h = tf.reshape(scale[0] * tf.sqrt(ratio), (-1, 1))
        w = tf.reshape(scale[1] / tf.sqrt(ratio), (-1, 1))

        anchors = tf.concat([-0.5 * w, -0.5 * h, 0.5 * w, 0.5 * h], axis = -1)

        # Shift base anchor
        center_x = (tf.cast(tf.range(feature_shape[1]), tf.float32) + 0.5) * tf.cast(1 / feature_shape[1], tf.float32) * tf.cast(stride[1], tf.float32)
        center_y = (tf.cast(tf.range(feature_shape[0]), tf.float32) + 0.5) * tf.cast(1 / feature_shape[0], tf.float32) * tf.cast(stride[0], tf.float32)

        center_x, center_y = tf.meshgrid(center_x, center_y)
        center_x = tf.reshape(center_x, [-1])
        center_y = tf.reshape(center_y, [-1])

        shifts = tf.expand_dims(tf.stack([center_x, center_y, center_x, center_y], axis = 1), axis = 1)
        anchors = tf.cast(tf.expand_dims(anchors, axis = 0), tf.float32) + shifts
        shape = [-1, 4]
        if not flatten:
            shape = tf.concat([feature_shape, shape], axis = 0)
        anchors = tf.reshape(anchors, shape)
        out.append(tf.stop_gradient(anchors))
    if len(out) == 1:
        out = out[0]
    elif concat and flatten:
        out = tf.concat(out, axis = 0)
    return out
    
def generate_yolo_anchors(feature, image_shape = [608, 608], size = [[0.01645, 0.02138], [0.02632, 0.04934], [0.05428, 0.03783],
                                                                     [0.04934, 0.10033], [0.10197, 0.07401], [0.09704, 0.19572],
                                                                     [0.19079, 0.14803], [0.25658, 0.32566], [0.61349, 0.53618]], 
                          normalize = True, auto_size = True, flatten = True, concat = True):
    """
    feature = feature or [features] or shape or [shapes]
    size = anchor_size (w, h)
    """
    if tf.is_tensor(feature) or not isinstance(feature, list) or isinstance(feature[0], int):
        feature = [feature]
    if not isinstance(size, list): #only one val > s
        size = [[size, size]]
    elif not isinstance(size[0], list): #only one size > [w, h]
        size = [size]

    if not isinstance(size[0][0], list):
        if auto_size and (len(size) % len(feature)) == 0:
            size = tf.split(size, len(feature))
        else:
            size = [size] * len(feature)

    out = []
    for x, size in zip(feature, size):
        if not isinstance(size, list) and not (tf.is_tensor(size) and tf.keras.backend.ndim(size) != 0):
            size = [size]
        
        feature_shape = x
        if tf.is_tensor(x) and 2 < tf.keras.backend.ndim(x) or (not tf.is_tensor(x) and 2 < np.ndim(x)):
            feature_shape = tf.shape(x) if tf.keras.backend.int_shape(x)[-3] is None else tf.keras.backend.int_shape(x)
        if 2 < np.shape(feature_shape)[0]:
            feature_shape = feature_shape[-3:-1]
        
        stride = [1, 1]
        if 1 < tf.reduce_max(size):
            if normalize:
                shape = image_shape
                ndim = (tf.keras.backend.ndim(shape) if tf.is_tensor(shape) else np.ndim(shape)) - np.ndim(size)
                for _ in range(ndim):
                    shape = tf.expand_dims(shape, axis = -1)
                size = tf.divide(tf.cast(size, tf.float32), tf.cast(shape, tf.float32))
            else:
                stride = image_shape
        
        # Generate base anchor
        w, h = tf.split(tf.cast(size, tf.float32), 2, axis = -1)
        w = tf.reshape(w, [-1, 1])
        h = tf.reshape(h, [-1, 1])

        anchors = tf.concat([-0.5 * w, -0.5 * h, 0.5 * w, 0.5 * h], axis = -1)

        # Shift base anchor
        center_x = (tf.cast(tf.range(feature_shape[1]), tf.float32) + 0.5) * tf.cast(1 / feature_shape[1], tf.float32) * tf.cast(stride[1], tf.float32)
        center_y = (tf.cast(tf.range(feature_shape[0]), tf.float32) + 0.5) * tf.cast(1 / feature_shape[0], tf.float32) * tf.cast(stride[0], tf.float32)

        center_x, center_y = tf.meshgrid(center_x, center_y)
        center_x = tf.reshape(center_x, [-1])
        center_y = tf.reshape(center_y, [-1])

        shifts = tf.expand_dims(tf.stack([center_x, center_y, center_x, center_y], axis = 1), axis = 1)
        anchors = tf.cast(tf.expand_dims(anchors, axis = 0), tf.float32) + shifts
        shape = [-1, 4]
        if not flatten:
            shape = tf.concat([feature_shape, shape], axis = 0)
        anchors = tf.reshape(anchors, shape)
        out.append(tf.stop_gradient(anchors))
    if len(out) == 1:
        out = out[0]
    elif concat and flatten:
        out = tf.concat(out, axis = 0)
    return out

def generate_points(feature, image_shape = [1024, 1024], stride = None, normalize = True, flatten = True, concat = True):
    if tf.is_tensor(feature) or not isinstance(feature, list) or isinstance(feature[0], int):
        feature = [feature]
    if not isinstance(stride, list):
        stride = [stride] * len(feature)
    
    out = []
    for x, stride in zip(feature, stride):
        shape = x
        if tf.is_tensor(x) and 2 < tf.keras.backend.ndim(x) or (not tf.is_tensor(x) and 2 < np.ndim(x)):
            shape = tf.shape(x) if tf.keras.backend.int_shape(x)[-3] is None else tf.keras.backend.int_shape(x)
        if 2 < np.shape(shape)[0]:
            shape = shape[-3:-1]
        
        stride = [stride, stride]
        if stride[0] is None:
            stride = tf.divide(tf.cast(image_shape, tf.float32), tf.cast(shape, tf.float32))
            if normalize:
                stride = tf.where(tf.greater(stride, 1), tf.divide(stride, tf.cast(image_shape, tf.float32)), stride)
        elif normalize and 1 < tf.reduce_max(stride):
            stride = tf.divide(stride, tf.cast(image_shape, tf.float32))
        shift = tf.cast(tf.divide(stride, 2), tf.float32)
        shift = tf.where(tf.greater(stride[0], 1), tf.floor(shift), shift)
        
        x, y = tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]))
        point = tf.cast(tf.stack([x, y], axis = -1), tf.float32) * stride[::-1] + shift[::-1]
        if flatten:
            point = tf.reshape(point, [-1, 2])
        out.append(tf.stop_gradient(point))
    if len(out) == 1:
        out = out[0]
    elif concat and flatten:
        out = tf.concat(out, axis = 0)
    return out