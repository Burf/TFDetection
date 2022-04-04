import tensorflow as tf

from .common import smooth_l1_loss

def score_accuracy(match, score, threshold = 0.5, missing_value = 0.):
    """
    match = -1 : negative / 0 : neutral / 1 : positive #(batch_size, sampling_count, 1)
    score = score for FG/BG #(batch_size, sampling_count, 1)
    """
    match_score = tf.cast(tf.equal(match, 1), tf.int32)
    indices = tf.where(tf.not_equal(match, 0))
    score = tf.gather_nd(score, indices)
    match_score = tf.gather_nd(match_score, indices)
    
    match_score = tf.expand_dims(tf.cast(match_score, score.dtype), axis = -1)
    score = tf.expand_dims(tf.clip_by_value(score, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()), axis = -1)
    score = tf.cast(tf.greater_equal(score, threshold), score.dtype)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(match_score, score), tf.float32))
    accuracy = tf.where(tf.math.is_nan(accuracy), missing_value, accuracy)
    return accuracy

def score_loss(match, score, missing_value = 0.):
    """
    match = -1 : negative / 0 : neutral / 1 : positive #(batch_size, sampling_count, 1)
    score = score for FG/BG #(batch_size, sampling_count, 1)
    """
    match_score = tf.cast(tf.equal(match, 1), tf.int32)
    indices = tf.where(tf.not_equal(match, 0))
    score = tf.gather_nd(score, indices)
    match_score = tf.gather_nd(match_score, indices)
    
    match_score = tf.expand_dims(tf.cast(match_score, score.dtype), axis = -1)
    score = tf.expand_dims(tf.clip_by_value(score, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()), axis = -1)
    
    loss = tf.keras.losses.binary_crossentropy(match_score, score)
    loss = tf.reduce_mean(loss)
    loss = tf.where(tf.math.is_nan(loss), missing_value, loss)
    return loss

def logits_accuracy(y_true, y_pred, missing_value = 0.):
    """
    y_true = targeted label #(batch_size, sampling_count, 1 or num_class)
    y_pred = targeted logits  #(batch_size, sampling_count, num_class)
    """
    n_true_class = tf.shape(y_true)[-1]
    n_pred_class = tf.shape(y_pred)[-1]
    y_true = tf.reshape(y_true, (-1, n_true_class))
    y_pred = tf.reshape(y_pred, (-1, n_pred_class))
    
    pred_indices = tf.where(tf.reduce_max(tf.cast(0 < y_pred, tf.int32), axis = -1))
    y_true = tf.gather_nd(y_true, pred_indices)
    y_pred = tf.gather_nd(y_pred, pred_indices)
    
    y_true = tf.cond(tf.equal(n_true_class, 1), true_fn = lambda: y_true[..., 0], false_fn = lambda: tf.cast(tf.argmax(y_true, axis = -1), y_true.dtype))
    y_pred = tf.argmax(y_pred, axis = -1)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    accuracy = tf.where(tf.math.is_nan(accuracy), missing_value, accuracy)
    return accuracy

def logits_loss(y_true, y_pred, focal = True, alpha = 1., gamma = 2., weight = None, missing_value = 0.):
    """
    y_true = targeted label #(batch_size, sampling_count, 1 or num_class)
    y_pred = targeted logits  #(batch_size, sampling_count, num_class)
    """
    n_true_class = tf.shape(y_true)[-1]
    n_pred_class = tf.shape(y_pred)[-1]
    y_true = tf.reshape(y_true, (-1, n_true_class))
    y_pred = tf.reshape(y_pred, (-1, n_pred_class))
    
    pred_indices = tf.where(tf.reduce_max(tf.cast(0 < y_pred, tf.int32), axis = -1))
    y_true = tf.gather_nd(y_true, pred_indices)
    y_pred = tf.gather_nd(y_pred, pred_indices)
    
    y_true = tf.cond(tf.equal(n_true_class, 1), true_fn = lambda: tf.one_hot(tf.cast(y_true, tf.int32), n_pred_class)[:, 0], false_fn = lambda: y_true)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = y_pred / (tf.reduce_sum(y_pred, axis = -1, keepdims = True) + tf.keras.backend.epsilon())
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
    
    loss = -y_true * tf.math.log(y_pred)
    if focal:
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * loss
    if weight is not None:
        loss *= weight
    loss = tf.reduce_sum(loss, axis = -1)
    loss = tf.reduce_mean(loss)
    loss = tf.where(tf.math.is_nan(loss), missing_value, loss)
    return loss

def regress_loss(match_or_y_true, bbox_true, bbox_pred, sigma = 1, missing_value = 0.):
    """
    match_or_y_true = targeted rpn_match or y_true #(batch_size, sampling_count, 1 or num_class)
    bbox_true = targeted true bbox #(batch_size, sampling_count, delta)
    bbox_pred = targeted pred bbox #(batch_size, sampling_count, delta)
    """
    match_or_y_true = tf.cond(tf.equal(tf.shape(match_or_y_true)[-1], 1), true_fn = lambda: match_or_y_true, false_fn = lambda: tf.expand_dims(tf.cast(tf.argmax(match_or_y_true, -1), match_or_y_true.dtype), axis = -1))
    match_or_y_true = tf.reshape(match_or_y_true, (-1,))
    bbox_true = tf.reshape(bbox_true, (-1, 4))
    bbox_pred = tf.reshape(bbox_pred, (-1, 4))
    
    true_indices = tf.where(0 < match_or_y_true)
    bbox_true = tf.gather_nd(bbox_true, true_indices)
    bbox_pred = tf.gather_nd(bbox_pred, true_indices)
    
    loss = smooth_l1_loss(bbox_true, bbox_pred, sigma)
    loss = tf.reduce_mean(loss)
    loss = tf.where(tf.math.is_nan(loss), missing_value, loss)
    return loss

def mask_loss(y_true, mask_true, mask_pred, missing_value = 0.):
    """
    y_true = targeted y_true #(batch_size, sampling_count, 1 or num_class)
    mask_true = targeted true mask #(batch_size, sampling_count, h, w)
    mask_pred = targeted pred mask #(batch_size, sampling_count, h, w)
    """
    y_true = tf.cond(tf.equal(tf.shape(y_true)[-1], 1), true_fn = lambda: y_true, false_fn = lambda: tf.expand_dims(tf.cast(tf.argmax(y_true, -1), y_true.dtype), axis = -1))
    y_true = tf.reshape(y_true, (-1,))
    true_shape = tf.shape(mask_true)
    mask_true = tf.reshape(mask_true, (-1, true_shape[-2], true_shape[-1]))
    pred_shape = tf.shape(mask_pred)
    mask_pred = tf.reshape(mask_pred, (-1, pred_shape[-2], pred_shape[-1]))
    
    true_indices = tf.where(0 < y_true)
    mask_true = tf.gather_nd(mask_true, true_indices)
    mask_pred = tf.gather_nd(mask_pred, true_indices)
    
    loss = tf.keras.losses.binary_crossentropy(mask_true, mask_pred)
    loss = tf.reduce_mean(loss)
    loss = tf.where(tf.math.is_nan(loss), missing_value, loss)
    return loss

def semantic_loss(mask_true, semantic_pred):
    """
    mask_true = mask_true #(batch_size, padded_num_true, h, w)
    semantic_pred = semantic_regress #(batch_size, sampling_count, h, w)
    """
    semantic_true = tf.reduce_max(mask_true, axis = 1)
    if tf.keras.backend.ndim(semantic_true) == 3:
        semantic_true = tf.expand_dims(semantic_true, axis = -1)
    semantic_true = tf.image.resize(semantic_true, tf.shape(semantic_pred)[-3:-1], method = "bilinear")
    loss = tf.keras.losses.sparse_categorical_crossentropy(semantic_true, semantic_pred)
    loss = tf.reduce_mean(loss)
    return loss