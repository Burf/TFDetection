import tensorflow as tf

from tfdet.core.loss import binary_cross_entropy, categorical_cross_entropy, focal_categorical_cross_entropy, smooth_l1

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

    dtype = score.dtype
    score = tf.clip_by_value(score, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    score = tf.cast(tf.greater_equal(score, threshold), dtype)
  
    accuracy = tf.reduce_mean(tf.cast(tf.equal(match_score, score), dtype))
    accuracy = tf.where(tf.math.is_nan(accuracy), tf.cast(missing_value, dtype), accuracy)
    return accuracy

def score_loss(score_true, score_pred, loss = binary_cross_entropy, missing_value = 0.):
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
    
    _loss = loss(match_score, score, reduce = False)
    
    dtype = _loss.dtype
    true_count = tf.reduce_sum(match_score)
    _loss = tf.reduce_sum(_loss) / tf.maximum(tf.cast(true_count, dtype), tf.cast(1., dtype))
    _loss = tf.where(tf.math.is_nan(_loss), tf.cast(missing_value, dtype), _loss)
    return _loss

def logits_accuracy(y_true, y_pred, missing_value = 0.):
    """
    y_true = targeted label #(batch_size, sampling_count, 1 or num_class)
    y_pred = targeted logits  #(batch_size, sampling_count, num_class)
    """
    n_true_class = tf.shape(y_true)[-1]
    n_pred_class = tf.shape(y_pred)[-1]
    y_true = tf.reshape(y_true, (-1, n_true_class))
    y_pred = tf.reshape(y_pred, (-1, n_pred_class))
    
    #pred_indices = tf.where(0 < tf.reduce_max(y_pred, axis = -1))[:, 0]
    pred_indices = tf.where(tf.reduce_any(tf.greater(y_pred, 0), axis = -1))[:, 0]
    y_true = tf.gather(y_true, pred_indices)
    y_pred = tf.gather(y_pred, pred_indices)
    
    dtype = y_pred.dtype
    y_true = tf.cond(tf.equal(n_true_class, 1), true_fn = lambda: y_true[..., 0], false_fn = lambda: tf.cast(tf.argmax(y_true, axis = -1), y_true.dtype))
    y_pred = tf.argmax(y_pred, axis = -1)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), dtype))
    accuracy = tf.where(tf.math.is_nan(accuracy), tf.cast(missing_value, dtype), accuracy)
    return accuracy

def logits_loss(y_true, y_pred, loss = focal_categorical_cross_entropy, weight = None, missing_value = 0.):
    """
    y_true = targeted label #(batch_size, sampling_count, 1 or num_class)
    y_pred = targeted logits  #(batch_size, sampling_count, num_class)
    """
    n_true_class = tf.shape(y_true)[-1]
    n_pred_class = tf.shape(y_pred)[-1]
    y_true = tf.reshape(y_true, (-1, n_true_class))
    y_pred = tf.reshape(y_pred, (-1, n_pred_class))
    
    #pred_indices = tf.where(0 < tf.reduce_max(y_pred, axis = -1))[:, 0]
    pred_indices = tf.where(tf.reduce_any(tf.greater(y_pred, 0), axis = -1))[:, 0]
    y_true = tf.gather(y_true, pred_indices)
    y_pred = tf.gather(y_pred, pred_indices)
    
    y_true = tf.cond(tf.equal(n_true_class, 1), true_fn = lambda: tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), n_pred_class)[:, 0], y_true.dtype), false_fn = lambda: y_true)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    _loss = loss(y_true, y_pred, weight = weight, reduce = False)
    #_loss = tf.reduce_sum(_loss, axis = -1)
    
    dtype = _loss.dtype
    label = tf.argmax(y_true, axis = -1)
    true_count = tf.reduce_sum(tf.cast(0 < label, dtype))
    _loss = tf.reduce_sum(_loss) / tf.maximum(true_count, tf.cast(1., dtype))
    _loss = tf.where(tf.math.is_nan(_loss), tf.cast(missing_value, dtype), _loss)
    return _loss

def regress_loss(match_or_y_true, bbox_true, bbox_pred, loss = smooth_l1, missing_value = 0.):
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
    
    _loss = loss(bbox_true, bbox_pred, reduce = False)
    _loss = tf.reduce_sum(_loss, axis = -1)
    _loss = tf.reduce_mean(_loss)
    _loss = tf.where(tf.math.is_nan(_loss), tf.cast(missing_value, _loss.dtype), _loss)
    return _loss

def mask_loss(y_true, mask_true, mask_pred, loss = binary_cross_entropy, missing_value = 0.):
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
    
    _loss = loss(mask_true, mask_pred, reduce = False)
    _loss = tf.reduce_mean(_loss)
    _loss = tf.where(tf.math.is_nan(_loss), tf.cast(missing_value, _loss.dtype), _loss)
    return _loss

def semantic_loss(y_true, mask_true, semantic_pred, loss = categorical_cross_entropy, method = "bilinear", weight = None, missing_value = 0.):
    """
    y_true = targeted y_true #(batch_size, padded_num_true, 1 or num_class)
    mask_true = mask_true #(batch_size, padded_num_true, h, w, 1)
    semantic_pred = semantic_regress #(batch_size, h, w, n_class)
    """
    mask_shape = tf.shape(mask_true)
    semantic_shape = tf.shape(semantic_pred)
    y_true = tf.cond(tf.equal(tf.shape(y_true)[-1], 1), true_fn = lambda: tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), semantic_shape[-1])[..., 0, :], y_true.dtype), false_fn = lambda: y_true)
    semantic_true = tf.reshape(mask_true, [mask_shape[0] * mask_shape[1], mask_shape[2], mask_shape[3], 1])
    semantic_true = tf.image.resize(semantic_true, semantic_shape[-3:-1], method = method)
    semantic_true = tf.clip_by_value(tf.round(semantic_true), 0., 1.)
    semantic_true = tf.reshape(semantic_true, [mask_shape[0], mask_shape[1], semantic_shape[-3], semantic_shape[-2], 1])
    semantic_true = tf.multiply(tf.cast(tf.expand_dims(tf.expand_dims(y_true, axis = -2), axis = -2), semantic_true.dtype), semantic_true)
    semantic_true = tf.reduce_max(semantic_true, axis = 1)
    
    _loss = loss(semantic_true, semantic_pred, weight = weight, reduce = False)
 
    _loss = tf.reduce_sum(_loss, axis = -1)
    _loss = tf.reduce_mean(_loss)
    _loss = tf.where(tf.math.is_nan(_loss), tf.cast(missing_value, _loss.dtype), _loss)
    return _loss
