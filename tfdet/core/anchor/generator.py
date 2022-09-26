import tensorflow as tf
import numpy as np

def generate_anchors(feature, image_shape = [1024, 1024], scale = [32, 64, 128, 256, 512], ratio = [0.5, 1, 2], normalize = True, auto_scale = True, flatten = True, concat = True, dtype = tf.float32):
    """
    feature = feature or [features] or shape or [shapes]
    scale = anchor_size scale
    ratio = anchor_height * np.sqrt(ratio), anchor_weight / np.sqrt(ratio)
    """
    if tf.is_tensor(feature) or not isinstance(feature, list) or isinstance(feature[0], int):
        feature = [feature]
    
    if tf.is_tensor(image_shape) and 2 < tf.keras.backend.ndim(image_shape) or (not tf.is_tensor(image_shape) and 2 < np.ndim(image_shape)):
        image_shape = tf.shape(image_shape) if tf.keras.backend.int_shape(image_shape)[-3] is None else tf.keras.backend.int_shape(image_shape)
    if 2 < np.shape(image_shape)[0]:
        image_shape = image_shape[-3:-1]
        
    scale = np.squeeze(scale)
    if np.ndim(scale) == 0:
        scale = [[scale]]
    elif np.ndim(scale) == 1:
        scale = np.expand_dims(scale, axis = -1)
    
    if np.shape(scale)[-1] == 1:
        scale = np.squeeze(scale, axis = -1)
    if auto_scale:
        if np.ndim(scale) != 2:
            if (len(scale) % len(feature)) != 0:
                scale = [scale] * len(feature)
            else:
                scale = np.expand_dims(scale, axis = -1)
    
    if np.ndim(ratio) != 2:
        ratio = [([ratio] if np.ndim(ratio) == 0 else ratio)] * len(feature)

    out = []
    for x, scale, ratio in zip(feature, scale, ratio):
        #if np.ndim(scale) == 0 and not (tf.is_tensor(scale) and tf.keras.backend.ndim(scale) != 0):
        #    scale = [scale]
        scale = [scale, scale]
        
        feature_shape = x
        if tf.is_tensor(x) and 2 < tf.keras.backend.ndim(x) or (not tf.is_tensor(x) and 2 < np.ndim(x)):
            feature_shape = tf.shape(x) if tf.keras.backend.int_shape(x)[-3] is None else tf.keras.backend.int_shape(x)
        if 2 < np.shape(feature_shape)[0]:
            feature_shape = feature_shape[-3:-1]
        
        stride = [1, 1]
        if 2 <= tf.reduce_max(scale[0]):
            if normalize:
                #shape = image_shape
                #ndim = (tf.keras.backend.ndim(shape) if tf.is_tensor(shape) else np.ndim(shape)) - np.ndim(scale)
                #for _ in range(ndim):
                #    shape = tf.expand_dims(shape, axis = -1)
                #scale = tf.divide(tf.cast(scale, dtype), tf.cast(shape, dtype))
                scale = tf.divide(tf.cast(scale, dtype), tf.cast(tf.expand_dims(image_shape, axis = -1), dtype))
            else:
                stride = image_shape
        
        # Generate base anchor
        scale = tf.cast(tf.expand_dims(scale, axis = 1), dtype)
        ratio = tf.cast(tf.expand_dims(ratio, axis = 1), dtype)

        h = tf.reshape(scale[0] * tf.sqrt(ratio), (-1, 1))
        w = tf.reshape(scale[1] / tf.sqrt(ratio), (-1, 1))

        anchors = tf.concat([-0.5 * w, -0.5 * h, 0.5 * w, 0.5 * h], axis = -1)

        # Shift base anchor
        center_x = (tf.cast(tf.range(feature_shape[1]), dtype) + 0.5) * tf.cast(1 / feature_shape[1], dtype) * tf.cast(stride[1], dtype)
        center_y = (tf.cast(tf.range(feature_shape[0]), dtype) + 0.5) * tf.cast(1 / feature_shape[0], dtype) * tf.cast(stride[0], dtype)

        center_x, center_y = tf.meshgrid(center_x, center_y)
        center_x = tf.reshape(center_x, [-1])
        center_y = tf.reshape(center_y, [-1])

        shifts = tf.expand_dims(tf.stack([center_x, center_y, center_x, center_y], axis = 1), axis = 1)
        anchors = tf.cast(tf.expand_dims(anchors, axis = 0), dtype) + shifts
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
    
def generate_yolo_anchors(feature, image_shape = [608, 608], size = [[ 10, 13], [ 16,  30], [ 33,  23],
                                                                     [ 30, 61], [ 62,  45], [ 59, 119],
                                                                     [116, 90], [156, 198], [373, 326]],
                          normalize = True, auto_size = True, flatten = True, concat = True, dtype = tf.float32):
    """
    feature = feature or [features] or shape or [shapes]
    size = anchor_size (w, h)
    """
    if tf.is_tensor(feature) or not isinstance(feature, list) or isinstance(feature[0], int):
        feature = [feature]
    
    if tf.is_tensor(image_shape) and 2 < tf.keras.backend.ndim(image_shape) or (not tf.is_tensor(image_shape) and 2 < np.ndim(image_shape)):
        image_shape = tf.shape(image_shape) if tf.keras.backend.int_shape(image_shape)[-3] is None else tf.keras.backend.int_shape(image_shape)
    if 2 < np.shape(image_shape)[0]:
        image_shape = image_shape[-3:-1]
    
    size = np.squeeze(size)
    if np.ndim(size) == 0:
        size = [[size, size]]
    elif np.ndim(size) == 1:
        size = np.expand_dims(size, axis = -1)

    if np.shape(size)[-1] == 1:
        size = np.tile(size, [1, 2])
    if auto_size and (len(size) % len(feature)) != 0:
        size = np.tile(size, [len(feature), 1])
    size = np.split(size, len(feature))

    out = []
    for x, size in zip(feature, size):
        #if np.ndim(size) == 0 and not (tf.is_tensor(size) and tf.keras.backend.ndim(size) != 0):
        #    size = [size]
        
        feature_shape = x
        if tf.is_tensor(x) and 2 < tf.keras.backend.ndim(x) or (not tf.is_tensor(x) and 2 < np.ndim(x)):
            feature_shape = tf.shape(x) if tf.keras.backend.int_shape(x)[-3] is None else tf.keras.backend.int_shape(x)
        if 2 < np.shape(feature_shape)[0]:
            feature_shape = feature_shape[-3:-1]
        
        stride = [1, 1]
        if 2 <= tf.reduce_max(size):
            if normalize:
                #shape = image_shape
                #ndim = (tf.keras.backend.ndim(shape) if tf.is_tensor(shape) else np.ndim(shape)) - np.ndim(size)
                #for _ in range(ndim):
                #    shape = tf.expand_dims(shape, axis = -1)
                #size = tf.divide(tf.cast(size, dtype), tf.cast(shape, dtype))
                size = tf.divide(tf.cast(size, dtype), tf.cast(tf.expand_dims(image_shape, axis = 0), dtype))
            else:
                stride = image_shape
        
        # Generate base anchor
        w, h = tf.split(tf.cast(size, dtype), 2, axis = -1)
        w = tf.reshape(w, [-1, 1])
        h = tf.reshape(h, [-1, 1])

        anchors = tf.concat([-0.5 * w, -0.5 * h, 0.5 * w, 0.5 * h], axis = -1)

        # Shift base anchor
        center_x = (tf.cast(tf.range(feature_shape[1]), dtype) + 0.5) * tf.cast(1 / feature_shape[1], dtype) * tf.cast(stride[1], dtype)
        center_y = (tf.cast(tf.range(feature_shape[0]), dtype) + 0.5) * tf.cast(1 / feature_shape[0], dtype) * tf.cast(stride[0], dtype)

        center_x, center_y = tf.meshgrid(center_x, center_y)
        center_x = tf.reshape(center_x, [-1])
        center_y = tf.reshape(center_y, [-1])

        shifts = tf.expand_dims(tf.stack([center_x, center_y, center_x, center_y], axis = 1), axis = 1)
        anchors = tf.cast(tf.expand_dims(anchors, axis = 0), dtype) + shifts
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

def generate_points(feature, image_shape = [1024, 1024], stride = None, normalize = True, flatten = True, concat = True, dtype = tf.float32):
    if tf.is_tensor(feature) or not isinstance(feature, list) or isinstance(feature[0], int):
        feature = [feature]
    if np.ndim(stride) == 0:
        stride = [stride] * len(feature)
    
    if tf.is_tensor(image_shape) and 2 < tf.keras.backend.ndim(image_shape) or (not tf.is_tensor(image_shape) and 2 < np.ndim(image_shape)):
        image_shape = tf.shape(image_shape) if tf.keras.backend.int_shape(image_shape)[-3] is None else tf.keras.backend.int_shape(image_shape)
    if 2 < np.shape(image_shape)[0]:
        image_shape = image_shape[-3:-1]
    
    out = []
    for x, stride in zip(feature, stride):
        shape = x
        if tf.is_tensor(x) and 2 < tf.keras.backend.ndim(x) or (not tf.is_tensor(x) and 2 < np.ndim(x)):
            shape = tf.shape(x) if tf.keras.backend.int_shape(x)[-3] is None else tf.keras.backend.int_shape(x)
        if 2 < np.shape(shape)[0]:
            shape = shape[-3:-1]
        
        stride = [stride, stride]
        if stride[0] is None:
            stride = tf.divide(tf.cast(image_shape, dtype), tf.cast(shape, dtype))
            if normalize:
                stride = tf.where(tf.greater(stride, 1), tf.divide(stride, tf.cast(image_shape, dtype)), stride)
        elif normalize and 2 <= tf.reduce_max(stride):
            stride = tf.divide(stride, tf.cast(image_shape, dtype))
        shift = tf.cast(tf.divide(stride, 2), dtype)
        shift = tf.where(tf.greater(stride[0], 1), tf.floor(shift), shift)
        
        x, y = tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]))
        point = tf.cast(tf.stack([x, y], axis = -1), dtype) * stride[::-1] + shift[::-1]
        if flatten:
            point = tf.reshape(point, [-1, 2])
        out.append(tf.stop_gradient(point))
    if len(out) == 1:
        out = out[0]
    elif concat and flatten:
        out = tf.concat(out, axis = 0)
    return out