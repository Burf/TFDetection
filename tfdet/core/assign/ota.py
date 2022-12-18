import tensorflow as tf

from ..bbox import overlap_bbox, isin
from ..loss import binary_cross_entropy

def dynamic_k_match(cost_matrix, iou_matrix, k = 10, batch_size = 10):
    num_true = tf.keras.backend.int_shape(cost_matrix)[1] if tf.keras.backend.int_shape(cost_matrix)[1] is not None else tf.shape(cost_matrix)[1]
    match_matrix = tf.zeros_like(cost_matrix, dtype = tf.int32)
    top_iou = tf.transpose(tf.nn.top_k(tf.transpose(iou_matrix), k = tf.minimum(k, tf.shape(cost_matrix)[0])).values)
    dynamic_k = tf.cast(tf.maximum(tf.round(tf.reduce_sum(top_iou, axis = 0)), 1), tf.int32)
    negativ_cost_matrix = tf.negative(cost_matrix)
    if num_true != 0:
        def calculate(index, match_matrix):
            pos_indices = tf.nn.top_k(negativ_cost_matrix[:, index], k = dynamic_k[index]).indices
            pos_indices = tf.stack([pos_indices, tf.fill([tf.shape(pos_indices)[0]], index)], axis = 1)
            return tf.tensor_scatter_nd_update(match_matrix, pos_indices, tf.ones(tf.shape(pos_indices)[0], dtype = match_matrix.dtype))
        match_matrix = tf.while_loop(lambda index, match_matrix: index < num_true,
                                     lambda index, match_matrix: (index + 1, calculate(index, match_matrix)),
                                     (0, match_matrix),
                                     parallel_iterations = batch_size)[1]

        prior_match_flag = tf.greater(tf.reduce_sum(match_matrix, axis = 1), 1)
        if tf.greater(tf.reduce_sum(tf.cast(prior_match_flag, tf.int32)), 0):
            min_cost = tf.reduce_min(cost_matrix, axis = 1)
            min_cost = tf.where(prior_match_flag, min_cost, min_cost - 1)
            prior_match_flag = tf.tile(tf.expand_dims(prior_match_flag, axis = -1), [1, num_true])
            cost_flag = tf.equal(cost_matrix, tf.expand_dims(min_cost, axis = 1))
            match_matrix = tf.where(prior_match_flag, 0, match_matrix)
            match_matrix = tf.where(cost_flag, 1, match_matrix)
    return tf.cast(match_matrix, tf.bool)

def sim_ota(y_true, bbox_true, y_pred, bbox_pred, extra_length = None, k = 10, iou_weight = 3., class_weight = 1., cross_entropy = binary_cross_entropy, batch_size = 10, mode = "normal"):
    """
    https://github.com/Megvii-BaseDetection/YOLOX
    
    -extra_length
    norm_bbox > extra_length = 2.5 / image_size, unnorm_bbox > extra_length = 2.5
    
    -cross_entropy
    binary_cross_entropy = tfdet.core.loss.binary_cross_entropy
    focal_binary_cross_entropy = tfdet.core.loss.focal_binary_cross_entropy
    """
    isin_flag = tf.transpose(isin(bbox_true, bbox_pred, extra_length = extra_length, mode = "circle")) #(P, T)
    valid_flag = tf.cast(tf.reduce_max(tf.cast(isin_flag, tf.int32), axis = 1), tf.bool)
    valid_indices = tf.where(valid_flag)[:, 0]
    
    num_true = tf.keras.backend.int_shape(bbox_true)[1] if tf.keras.backend.int_shape(bbox_true)[1] is not None else tf.shape(bbox_true)[1]
    num_pred = tf.keras.backend.int_shape(valid_indices)[0] if tf.keras.backend.int_shape(valid_indices)[0] is not None else tf.shape(valid_indices)[0]
    if num_true != 0 and num_pred != 0:
        y_pred = tf.gather(y_pred, valid_indices)
        bbox_pred = tf.gather(bbox_pred, valid_indices)
        valid_isin_flag = tf.gather(isin_flag, valid_indices)

        overlaps = overlap_bbox(bbox_pred, bbox_true, mode = mode) #(P, T)
        iou_cost = tf.negative(overlaps + tf.keras.backend.epsilon())

        true_count = tf.shape(y_true)[0]
        pred_count = tf.shape(y_pred)[0]
        y_true = tf.reshape(tf.tile(y_true, [pred_count, 1]), [pred_count, true_count, -1])
        y_pred = tf.reshape(tf.tile(y_pred, [1, true_count]), [pred_count, true_count, -1])

        class_cost = cross_entropy(y_true, tf.sqrt(y_pred), reduce = False)
        class_cost = tf.reduce_sum(class_cost, axis = -1)

        inf = 100000.
        cost_matrix = class_cost * class_weight + iou_cost * iou_weight + tf.cast(tf.logical_not(valid_isin_flag), tf.float32) * inf
        match_matrix = dynamic_k_match(cost_matrix, overlaps, k = k, batch_size = batch_size)

        match = tf.reduce_max(tf.cast(match_matrix, tf.int32), axis = 1)
        match_indices = tf.where(match == 1)[:, 0]
        match_negative_indices = tf.where(match != 1)[:, 0]
        positive_indices = tf.gather(valid_indices, match_indices)
        negative_indices = tf.gather(valid_indices, match_negative_indices)
        valid_negative_indices = tf.where(tf.logical_not(valid_flag))[:, 0]
        negative_indices = tf.sort(tf.concat([negative_indices, valid_negative_indices], axis = 0))
        positive_matrix = tf.gather(match_matrix, match_indices)
        true_indices = tf.cond(tf.greater(tf.shape(positive_matrix)[1], 0), true_fn = lambda: tf.argmax(positive_matrix, axis = -1), false_fn = lambda: tf.cast(tf.constant([]), tf.int64))
    else:
        true_indices = positive_indices = valid_indices
        negative_indices = tf.where(tf.logical_not(valid_flag))[:, 0]
    return true_indices, positive_indices, negative_indices

def align_ota(y_true, bbox_true, y_pred, bbox_pred, extra_length = None, k = 10, iou_weight = 3., class_weight = 1., cross_entropy = binary_cross_entropy, batch_size = 10, mode = "normal"):
    """
    https://github.com/tinyvision/DAMO-YOLO
    
    -extra_length
    norm_bbox > extra_length = 2.5 / image_size, unnorm_bbox > extra_length = 2.5
    
    -cross_entropy
    binary_cross_entropy = tfdet.core.loss.binary_cross_entropy
    focal_binary_cross_entropy = tfdet.core.loss.focal_binary_cross_entropy
    """
    isin_flag = tf.transpose(isin(bbox_true, bbox_pred, extra_length = extra_length, mode = "circle")) #(P, T)
    valid_flag = tf.cast(tf.reduce_max(tf.cast(isin_flag, tf.int32), axis = 1), tf.bool)
    valid_indices = tf.where(valid_flag)[:, 0]
    
    num_true = tf.keras.backend.int_shape(bbox_true)[1] if tf.keras.backend.int_shape(bbox_true)[1] is not None else tf.shape(bbox_true)[1]
    num_pred = tf.keras.backend.int_shape(valid_indices)[0] if tf.keras.backend.int_shape(valid_indices)[0] is not None else tf.shape(valid_indices)[0]
    if num_true != 0 and num_pred != 0:
        y_pred = tf.gather(y_pred, valid_indices)
        bbox_pred = tf.gather(bbox_pred, valid_indices)
        valid_isin_flag = tf.gather(isin_flag, valid_indices)

        overlaps = overlap_bbox(bbox_pred, bbox_true, mode = mode) #(P, T)
        iou_cost = tf.negative(overlaps + tf.keras.backend.epsilon())

        true_count = tf.shape(y_true)[0]
        pred_count = tf.shape(y_pred)[0]
        y_true = tf.cast(tf.cond(tf.equal(tf.shape(y_true)[-1], 1), true_fn = lambda: tf.one_hot(tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1])[:, 0], false_fn = lambda: y_true), y_pred.dtype)
        y_true = tf.reshape(tf.tile(y_true, [pred_count, 1]), [pred_count, true_count, -1])
        y_pred = tf.reshape(tf.tile(y_pred, [1, true_count]), [pred_count, true_count, -1])
        
        soft_y_true = y_true * tf.expand_dims(overlaps, axis = -1)
        scale_factor = soft_y_true - y_pred

        class_cost = cross_entropy(soft_y_true, y_pred, reduce = False)
        class_cost = class_cost * tf.pow(tf.abs(scale_factor), 2)
        class_cost = tf.reduce_sum(class_cost, axis = -1)

        inf = 100000.
        cost_matrix = class_cost * class_weight + iou_cost * iou_weight + tf.cast(tf.logical_not(valid_isin_flag), tf.float32) * inf
        match_matrix = dynamic_k_match(cost_matrix, overlaps, k = k, batch_size = batch_size)

        match = tf.reduce_max(tf.cast(match_matrix, tf.int32), axis = 1)
        match_indices = tf.where(match == 1)[:, 0]
        match_negative_indices = tf.where(match != 1)[:, 0]
        positive_indices = tf.gather(valid_indices, match_indices)
        negative_indices = tf.gather(valid_indices, match_negative_indices)
        valid_negative_indices = tf.where(tf.logical_not(valid_flag))[:, 0]
        negative_indices = tf.sort(tf.concat([negative_indices, valid_negative_indices], axis = 0))
        positive_matrix = tf.gather(match_matrix, match_indices)
        true_indices = tf.cond(tf.greater(tf.shape(positive_matrix)[1], 0), true_fn = lambda: tf.argmax(positive_matrix, axis = -1), false_fn = lambda: tf.cast(tf.constant([]), tf.int64))
    else:
        true_indices = positive_indices = valid_indices
        negative_indices = tf.where(tf.logical_not(valid_flag))[:, 0]
    return true_indices, positive_indices, negative_indices