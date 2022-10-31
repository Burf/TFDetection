import tensorflow as tf

from tfdet.core.loss import smooth_l1

def score_accuracy(score_true, score_pred, threshold = 0.5, missing_value = 0.):
    """
    score_true = -1 : negative / 0 : neutral / 1 : positive #(batch_size, sampling_count, 1)
    score_pred = score for FG/BG #(batch_size, sampling_count, 1)
    """
    score_true = tf.reshape(score_true, (-1, 1))
    score_pred = tf.reshape(score_pred, (-1, 1))
    
    indices = tf.where(tf.equal(score_true, 1))[:, 0]
    score = tf.gather(score_pred, indices)
    match_score = tf.ones_like(score)

    score = tf.clip_by_value(score, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    score = tf.cast(tf.greater_equal(score, threshold), score.dtype)
  
    accuracy = tf.reduce_mean(tf.cast(tf.equal(match_score, score), score.dtype))
    accuracy = tf.where(tf.math.is_nan(accuracy), missing_value, accuracy)
    return accuracy

def score_loss(score_true, score_pred, focal = True, missing_value = 0.):
    """
    score_true = -1 : negative / 0 : neutral / 1 : positive #(batch_size, sampling_count, 1)
    score_pred = score for FG/BG #(batch_size, sampling_count, 1)
    """
    score_true = tf.reshape(score_true, (-1, 1))
    score_pred = tf.reshape(score_pred, (-1, 1))
    
    match_score = tf.cast(tf.equal(score_true, 1), tf.int32)
    indices = tf.where(tf.not_equal(score_true, 0))[:, 0]
    score = tf.gather(score_pred, indices)
    match_score = tf.gather(match_score, indices)

    match_score = tf.cast(match_score, score_pred.dtype)
    score = tf.clip_by_value(score, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
  
    loss = tf.keras.losses.binary_crossentropy(match_score, score)
    if focal:
        loss = tf.expand_dims(loss, axis = -1) * tf.pow(match_score - score, 2)
    
    true_count = tf.reduce_sum(match_score)
    loss = tf.reduce_sum(loss) / tf.maximum(true_count, 1.)
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
    
    pred_indices = tf.where(0 < tf.reduce_max(y_pred, axis = -1))[:, 0]
    y_true = tf.gather(y_true, pred_indices)
    y_pred = tf.gather(y_pred, pred_indices)
    
    dtype = y_pred.dtype
    y_true = tf.cond(tf.equal(n_true_class, 1), true_fn = lambda: y_true[..., 0], false_fn = lambda: tf.cast(tf.argmax(y_true, axis = -1), y_true.dtype))
    y_pred = tf.argmax(y_pred, axis = -1)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), dtype))
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
    
    pred_indices = tf.where(0 < tf.reduce_max(y_pred, axis = -1))[:, 0]
    y_true = tf.gather(y_true, pred_indices)
    y_pred = tf.gather(y_pred, pred_indices)
    
    y_true = tf.cond(tf.equal(n_true_class, 1), true_fn = lambda: tf.one_hot(tf.cast(y_true, tf.int32), n_pred_class)[:, 0], false_fn = lambda: y_true)
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = y_pred / (tf.reduce_sum(y_pred, axis = -1, keepdims = True) + tf.keras.backend.epsilon())
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
    
    loss = -y_true * tf.math.log(y_pred)
    if focal:
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * loss
    if weight is not None:
        loss *= weight
    #loss = tf.reduce_sum(loss, axis = -1)
    
    label = tf.argmax(y_true, axis = -1)
    true_count = tf.reduce_sum(tf.cast(0 < label, y_pred.dtype))
    loss = tf.reduce_sum(loss) / tf.maximum(true_count, 1.)
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
    
    true_indices = tf.where(0 < match_or_y_true)[:, 0]
    bbox_true = tf.gather(bbox_true, true_indices)
    bbox_pred = tf.gather(bbox_pred, true_indices)
    
    loss = smooth_l1(bbox_true, bbox_pred, sigma)
    loss = tf.reduce_sum(loss, axis = -1)
    
    loss = tf.reduce_mean(loss)
    loss = tf.where(tf.math.is_nan(loss), missing_value, loss)
    return loss

def mask_loss(y_true, mask_true, mask_pred, missing_value = 0.):
    """
    y_true = targeted y_true #(batch_size, sampling_count, 1 or num_class)
    mask_true = targeted true mask #(batch_size, sampling_count, h, w, 1)
    mask_pred = targeted pred mask #(batch_size, sampling_count, h, w, 1)
    """
    y_true = tf.cond(tf.equal(tf.shape(y_true)[-1], 1), true_fn = lambda: y_true, false_fn = lambda: tf.expand_dims(tf.cast(tf.argmax(y_true, -1), y_true.dtype), axis = -1))
    y_true = tf.reshape(y_true, (-1,))
    batch = tf.shape(y_true)[0]
    
    mask_true = tf.reshape(mask_true, (batch, -1, 1))
    mask_pred = tf.reshape(mask_pred, (batch, -1, 1))
    
    true_indices = tf.where(0 < y_true)[:, 0]
    mask_true = tf.gather(mask_true, true_indices)
    mask_pred = tf.gather(mask_pred, true_indices)
    
    loss = tf.keras.losses.binary_crossentropy(mask_true, mask_pred)
    loss = tf.reduce_mean(loss)
    loss = tf.where(tf.math.is_nan(loss), missing_value, loss)
    return loss

def semantic_loss(y_true, mask_true, semantic_pred, method = "bilinear", weight = None, missing_value = 0.):
    """
    y_true = targeted y_true #(batch_size, padded_num_true, 1 or num_class)
    mask_true = mask_true #(batch_size, padded_num_true, h, w, 1)
    semantic_pred = semantic_regress #(batch_size, h, w, n_class)
    """
    mask_shape = tf.shape(mask_true)
    semantic_shape = tf.shape(semantic_pred)
    y_true = tf.cond(tf.equal(tf.shape(y_true)[-1], 1), true_fn = lambda: tf.one_hot(tf.cast(y_true, tf.int32), semantic_shape[-1])[..., 0, :], false_fn = lambda: y_true)
    semantic_true = tf.reshape(mask_true, [mask_shape[0] * mask_shape[1], mask_shape[2], mask_shape[3], 1])
    semantic_true = tf.image.resize(semantic_true, semantic_shape[-3:-1], method = method)
    semantic_true = tf.clip_by_value(tf.round(semantic_true), 0., 1.)
    semantic_true = tf.reshape(semantic_true, [mask_shape[0], mask_shape[1], semantic_shape[-3], semantic_shape[-2], 1])
    semantic_true = tf.multiply(tf.cast(tf.expand_dims(tf.expand_dims(y_true, axis = -2), axis = -2), semantic_true.dtype), semantic_true)
    semantic_true = tf.reduce_max(semantic_true, axis = 1)
    
    semantic_pred = semantic_pred / (tf.reduce_sum(semantic_pred, axis = -1, keepdims = True) + tf.keras.backend.epsilon())
    semantic_pred = tf.clip_by_value(semantic_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())

    loss = -tf.cast(semantic_true, semantic_pred.dtype) * tf.math.log(semantic_pred)
    if weight is not None:
        loss *= weight

    loss = tf.reduce_sum(loss, axis = -1)
    loss = tf.reduce_mean(loss)
    loss = tf.where(tf.math.is_nan(loss), missing_value, loss)
    return loss
