import tensorflow as tf

from tfdet.core.loss import binary_cross_entropy, giou

def giou_loss(bbox_true, bbox_pred, reduce = True, mode = "general"):
    bbox_true = tf.reshape(bbox_true, (-1, 4))
    bbox_pred = tf.reshape(bbox_pred, (-1, 4))
    
    loss = giou(bbox_true, bbox_pred, reduce = False, mode = mode)
    bbox_loss_scale = 1. - ((bbox_true[..., 2] - bbox_true[..., 0]) * (bbox_true[..., 3] - bbox_true[..., 1])) #2 - 1 * bbox_area / input_area
    loss = bbox_loss_scale * loss
    if reduce:
        loss = tf.reduce_mean(loss)
    return loss

def score_accuracy(score_true, score_pred, threshold = 0.5, missing_value = 0.):
    """
    score_true = -1 : negative / 0 : neutral / 1 : positive #(batch_size, sampling_count, 1)
    score_pred = confidence score for FG/BG #(batch_size, sampling_count, 1)
    """
    score_true = tf.reshape(score_true, (-1, 1))
    score_pred = tf.reshape(score_pred, (-1, 1))
    
    indices = tf.where(tf.equal(score_true, 1))[:, 0]
    score = tf.gather(score_pred, indices)
    match_score = tf.ones_like(score)
    match_score = tf.cast(match_score, score_pred.dtype)

    score = tf.clip_by_value(score, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    score = tf.cast(tf.greater_equal(score, threshold), score.dtype)
  
    accuracy = tf.reduce_mean(tf.cast(tf.equal(match_score, score), score.dtype))
    accuracy = tf.where(tf.math.is_nan(accuracy), missing_value, accuracy)
    return accuracy

def score_loss(score_true, score_pred, loss = binary_cross_entropy, missing_value = 0.):
    """
    score_true = -1 : negative / 0 : neutral / 1 : positive #(batch_size, sampling_count, 1)
    score_pred = confidence score for FG/BG #(batch_size, sampling_count, 1)
    """
    score_true = tf.reshape(score_true, (-1, 1))
    score_pred = tf.reshape(score_pred, (-1, 1))
    
    match_score = tf.cast(tf.equal(score_true, 1), tf.int32)
    indices = tf.where(tf.not_equal(score_true, 0))[:, 0]
    score = tf.gather(score_pred, indices)
    match_score = tf.gather(match_score, indices)

    _loss = loss(match_score, score, reduce = False)
    
    true_count = tf.reduce_sum(match_score)
    _loss = tf.reduce_sum(_loss) / tf.maximum(true_count, 1.)
    _loss = tf.where(tf.math.is_nan(_loss), missing_value, _loss)
    return _loss
    
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
    
    true_indices = tf.where(0 < score_true)[:, 0]
    logit_true = tf.gather(logit_true, true_indices)
    logit_pred = tf.gather(logit_pred, true_indices)

    dtype = logit_pred.dtype
    logit_true = tf.cond(tf.equal(n_true_class, 1), true_fn = lambda: logit_true[..., 0], false_fn = lambda: tf.cast(tf.argmax(logit_true, axis = -1), logit_true.dtype))
    logit_pred = tf.argmax(logit_pred, axis = -1)
    logit_true = tf.cast(logit_true, logit_pred.dtype)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(logit_true, logit_pred), dtype))
    accuracy = tf.where(tf.math.is_nan(accuracy), missing_value, accuracy)
    return accuracy

def logits_loss(score_true, logit_true, logit_pred, loss = binary_cross_entropy, weight = None, missing_value = 0.):
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
    
    true_indices = tf.where(0 < score_true)[:, 0]
    logit_true = tf.gather(logit_true, true_indices)
    logit_pred = tf.gather(logit_pred, true_indices)

    logit_true = tf.cond(tf.equal(n_true_class, 1), true_fn = lambda: tf.one_hot(tf.cast(logit_true, tf.int32), n_pred_class)[:, 0], false_fn = lambda: logit_true)
    logit_true = tf.cast(logit_true, logit_pred.dtype)
    logit_pred = tf.clip_by_value(logit_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    _loss = loss(logit_true, logit_pred, weight = weight, reduce = False)
    _loss = tf.reduce_sum(_loss, axis = -1)
    
    _loss = tf.reduce_mean(_loss)
    _loss = tf.where(tf.math.is_nan(_loss), missing_value, _loss)
    return _loss

def regress_loss(score_true, bbox_true, bbox_pred, loss = giou_loss, missing_value = 0.):
    """
    score_true = targeted score_true #(batch_size, sampling_count, 1)
    bbox_true = targeted true bbox #(batch_size, sampling_count, delta)
    bbox_pred = targeted pred bbox #(batch_size, sampling_count, delta)
    """
    score_true = tf.reshape(score_true, (-1,))
    bbox_true = tf.reshape(bbox_true, (-1, 4))
    bbox_pred = tf.reshape(bbox_pred, (-1, 4))
    
    true_indices = tf.where(0 < score_true)[:, 0]
    bbox_true = tf.gather(bbox_true, true_indices)
    bbox_pred = tf.gather(bbox_pred, true_indices)

    _loss = loss(bbox_true, bbox_pred, reduce = False)

    _loss = tf.reduce_mean(_loss)
    _loss = tf.where(tf.math.is_nan(_loss), missing_value, _loss)
    return _loss