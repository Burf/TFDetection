import tensorflow as tf

def roi2level(bbox, n_level, input_shape = (224, 224)):
    if 2 <= tf.reduce_max(bbox):
        bbox = tf.divide(bbox, tf.cast(tf.tile(input_shape[::-1], [2]), bbox.dtype))
    x1, y1, x2, y2 = tf.split(bbox, 4, axis = -1)
    h = y2 - y1
    w = x2 - x1

    bbox_area = h * w
    image_area = tf.cast(input_shape[0] * input_shape[1], bbox.dtype)

    roi_level = tf.cast(tf.floor(tf.math.log((tf.sqrt(bbox_area)) / ((56. / tf.sqrt(image_area)) + 1e-6)) / tf.math.log(2.)), tf.int32)
    roi_level = tf.clip_by_value(roi_level, 0, n_level - 1)
    roi_level = tf.squeeze(roi_level, axis = -1)
    return roi_level

def roi_align(feature, bbox_pred, image_shape = [1024, 1024], pool_size = 7, method = "bilinear"):
    if not isinstance(feature, list):
        feature = [feature]
    pool_size = [pool_size, pool_size] if isinstance(pool_size, int) else [pool_size, pool_size]
    
    max_size = tf.shape(bbox_pred)[0]
    valid_indices = tf.where(0 < tf.reduce_max(bbox_pred, axis = -1))[:, 0]
    bbox_pred = tf.gather(bbox_pred, valid_indices)
    
    if 2 <= tf.reduce_max(bbox_pred):
        bbox_pred = tf.divide(bbox_pred, tf.cast(tf.tile(image_shape[::-1], [2]), bbox_pred.dtype))
    
    roi_level = roi2level(bbox_pred, len(feature), image_shape)
    x1, y1, x2, y2 = tf.split(bbox_pred, 4, axis = -1)
    bbox_pred = tf.concat([y1, x1, y2, x2], axis = -1)
    
    indices = []
    result = []
    for level, x in enumerate(feature):
        level_indices = tf.where(tf.equal(roi_level, level))[:, 0]
        bbox = tf.gather(bbox_pred, level_indices)

        bbox = tf.stop_gradient(bbox)
        bbox_indices = tf.stop_gradient(tf.zeros(tf.shape(level_indices)[0], tf.int32))
        out = tf.image.crop_and_resize(image = tf.expand_dims(x, axis = 0), boxes = bbox, box_indices = bbox_indices, crop_size = pool_size, method = method)

        indices.append(level_indices)
        result.append(out)
    indices = tf.concat(indices, axis = 0)
    result = tf.concat(result, axis = 0)
    
    sorted_indices = tf.nn.top_k(indices, k = tf.shape(indices)[0]).indices[::-1]
    indices = tf.gather(indices, sorted_indices)
    result = tf.gather(result, indices)
    
    pad_size = max_size - tf.shape(bbox_pred)[0]
    result = tf.pad(result, [[0, pad_size], [0, 0], [0, 0], [0, 0]])
    result = tf.reshape(result, [max_size, *pool_size, tf.shape(feature[0])[-1]])
    return result