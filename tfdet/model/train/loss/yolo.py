import tensorflow as tf

from tfdet.core.bbox import overlap_bbox

def score_accuracy(score_true, score_pred, threshold = 0.5, missing_value = 0.):
    """
    score_true = -1 : negative / 0 : neutral / 1 : positive #(batch_size, sampling_count, 1)
    score_pred = confidence score for FG/BG #(batch_size, sampling_count, 1)
    """
    match_score = tf.cast(tf.equal(score_true, 1), tf.int32)
    indices = tf.where(tf.not_equal(score_true, 0))
    score = tf.gather_nd(score_pred, indices)
    match_score = tf.gather_nd(match_score, indices)

    match_score = tf.expand_dims(tf.cast(match_score, score_pred.dtype), axis = -1)
    score = tf.expand_dims(tf.clip_by_value(score, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()), axis = -1)
    score = tf.cast(tf.greater_equal(score, threshold), score.dtype)
  
    accuracy = tf.reduce_mean(tf.cast(tf.equal(match_score, score), score.dtype))
    accuracy = tf.where(tf.math.is_nan(accuracy), missing_value, accuracy)
    return accuracy

def score_loss(score_true, score_pred, focal = True, missing_value = 0.):
    """
    score_true = -1 : negative / 0 : neutral / 1 : positive #(batch_size, sampling_count, 1)
    score_pred = confidence score for FG/BG #(batch_size, sampling_count, 1)
    """
    match_score = tf.cast(tf.equal(score_true, 1), tf.int32)
    indices = tf.where(tf.not_equal(score_true, 0))
    score = tf.gather_nd(score_pred, indices)
    match_score = tf.gather_nd(match_score, indices)

    match_score = tf.expand_dims(tf.cast(match_score, score_pred.dtype), axis = -1)
    score = tf.expand_dims(tf.clip_by_value(score, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()), axis = -1)
  
    loss = tf.keras.losses.binary_crossentropy(match_score, score)
    if focal:
        loss = tf.expand_dims(loss, axis = -1) * tf.pow(match_score - score, 2)
    
    loss = tf.reduce_mean(loss)
    loss = tf.where(tf.math.is_nan(loss), missing_value, loss)
    return loss
    
def logits_accuracy(score_true, logit_true, logit_pred, missing_value = 0.):
    """
    score_true = targeted score_true #(batch_size, sampling_count, 1)
    logit_true = targeted label #(batch_size, sampling_count, 1 or num_class)
    logit_pred = targeted logits  #(batch_size, sampling_count, num_class)
    """
    n_true_class = tf.shape(logit_true)[-1]
    n_pred_class = tf.shape(logit_pred)[-1]
    score_true = tf.reshape(score_true, (-1,))
    logit_true = tf.reshape(logit_true, (-1, n_true_class))
    logit_pred = tf.reshape(logit_pred, (-1, n_pred_class))
    
    true_indices = tf.where(0 < score_true)
    logit_true = tf.gather_nd(logit_true, true_indices)
    logit_pred = tf.gather_nd(logit_pred, true_indices)

    dtype = logit_pred.dtype
    logit_true = tf.cond(tf.equal(n_true_class, 1), true_fn = lambda: logit_true[..., 0], false_fn = lambda: tf.cast(tf.argmax(logit_true, axis = -1), logit_true.dtype))
    logit_pred = tf.argmax(logit_pred, axis = -1)
    logit_true = tf.cast(logit_true, logit_pred.dtype)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(logit_true, logit_pred), dtype))
    accuracy = tf.where(tf.math.is_nan(accuracy), missing_value, accuracy)
    return accuracy

def logits_loss(score_true, logit_true, logit_pred, focal = True, alpha = .25, gamma = 1.5, weight = None, missing_value = 0.):
    """
    score_true = targeted score_true #(batch_size, sampling_count, 1)
    logit_true = targeted label #(batch_size, sampling_count, 1 or num_class)
    logit_pred = targeted logits  #(batch_size, sampling_count, num_class)
    """
    n_true_class = tf.shape(logit_true)[-1]
    n_pred_class = tf.shape(logit_pred)[-1]
    score_true = tf.reshape(score_true, (-1,))
    logit_true = tf.reshape(logit_true, (-1, n_true_class))
    logit_pred = tf.reshape(logit_pred, (-1, n_pred_class))
    
    true_indices = tf.where(0 < score_true)
    logit_true = tf.gather_nd(logit_true, true_indices)
    logit_pred = tf.gather_nd(logit_pred, true_indices)

    logit_true = tf.cond(tf.equal(n_true_class, 1), true_fn = lambda: tf.one_hot(tf.cast(logit_true, tf.int32), n_pred_class)[:, 0], false_fn = lambda: logit_true)
    logit_true = tf.cast(logit_true, logit_pred.dtype)
    logit_pred = tf.clip_by_value(logit_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    loss = -tf.stack([(1 - logit_true) * tf.math.log(1 - logit_pred), logit_true * tf.math.log(logit_pred)], axis = -1)
    if focal:
        alpha_factor = tf.ones_like(logit_true) * alpha
        alpha_factor = tf.where(0.5 < logit_true, alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(0.5 < logit_true, 1 - logit_pred, logit_pred)
        focal_weight = alpha_factor * focal_weight ** gamma
        loss = tf.expand_dims(focal_weight, axis = -1) * loss
    if weight is not None:
        loss *= weight
    loss = tf.reduce_sum(loss, axis = -1)
    loss = tf.reduce_mean(loss, axis = -1)
    
    loss = tf.reduce_mean(loss)
    loss = tf.where(tf.math.is_nan(loss), missing_value, loss)
    return loss

def regress_loss(score_true, bbox_true, bbox_pred, mode = "general", missing_value = 0.):
    """
    score_true = targeted score_true #(batch_size, sampling_count, 1)
    bbox_true = targeted true bbox #(batch_size, sampling_count, delta)
    bbox_pred = targeted pred bbox #(batch_size, sampling_count, delta)
    """
    score_true = tf.reshape(score_true, (-1,))
    bbox_true = tf.reshape(bbox_true, (-1, 4))
    bbox_pred = tf.reshape(bbox_pred, (-1, 4))
    
    true_indices = tf.where(0 < score_true)
    bbox_true = tf.gather_nd(bbox_true, true_indices)
    bbox_pred = tf.gather_nd(bbox_pred, true_indices)

    overlaps = overlap_bbox(bbox_true, bbox_pred, mode = mode) #(T, P)
    max_iou = tf.reduce_max(overlaps, axis = 0)
    bbox_loss_scale = 1. - ((bbox_true[..., 2] - bbox_true[..., 0]) * (bbox_true[..., 3] - bbox_true[..., 1])) #2 - 1 * bbox_area / input_area
    loss = bbox_loss_scale * (1 - max_iou)

    loss = tf.reduce_mean(loss)
    loss = tf.where(tf.math.is_nan(loss), missing_value, loss)
    return loss