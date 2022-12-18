import tensorflow as tf

from tfdet.core.loss import focal_binary_cross_entropy, smooth_l1

def focal_loss(y_true, y_pred, alpha = .25, gamma = 1.5, weight = None, reduce = True):
    return focal_binary_cross_entropy(y_true, y_pred, alpha = alpha, gamma = gamma, weight = weight, reduce = reduce)
    
def classnet_accuracy(y_true, y_pred, missing_value = 0.):
    """
    y_true = targeted label #(batch_size, sampling_count, 1 or num_class)
    y_pred = targeted logits  #(batch_size, sampling_count, num_class)
    """
    n_true_class = tf.shape(y_true)[-1]
    n_pred_class = tf.shape(y_pred)[-1]
    y_true = tf.reshape(y_true, (-1, n_true_class))
    y_pred = tf.reshape(y_pred, (-1, n_pred_class))
    
    y_true = tf.cond(tf.equal(n_true_class, 1), true_fn = lambda: y_true[..., 0], false_fn = lambda: tf.cast(tf.argmax(y_true, axis = -1), y_true.dtype))
    true_indices = tf.where(0 < y_true)[:, 0]
    y_true = tf.gather(y_true, true_indices)
    y_pred = tf.gather(y_pred, true_indices)
    
    dtype = y_pred.dtype
    y_pred = tf.argmax(y_pred, axis = -1)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), dtype))
    accuracy = tf.where(tf.math.is_nan(accuracy), missing_value, accuracy)
    return accuracy

def classnet_loss(y_true, y_pred, loss = focal_loss, weight = None, background = False, missing_value = 0.):
    """
    y_true = targeted label #(batch_size, sampling_count, 1 or num_class)
    y_pred = targeted logits  #(batch_size, sampling_count, num_class)
    """
    n_true_class = tf.shape(y_true)[-1]
    n_pred_class = tf.shape(y_pred)[-1]
    y_true = tf.reshape(y_true, (-1, n_true_class))
    y_pred = tf.reshape(y_pred, (-1, n_pred_class))
    
    if background:
        pred_indices = tf.where(0 < tf.reduce_max(y_pred, axis = -1))[:, 0]
        y_true = tf.gather(y_true, pred_indices)
        y_pred = tf.gather(y_pred, pred_indices)
    
    y_true = tf.cond(tf.equal(n_true_class, 1), true_fn = lambda: tf.one_hot(tf.cast(y_true, tf.int32), n_pred_class)[:, 0], false_fn = lambda: y_true)
    true_flag = tf.not_equal(tf.expand_dims(tf.argmax(y_true, axis = -1), axis = -1), 0)
    if not background:
        y_true = tf.where(true_flag, y_true, 0)
        
    _loss = loss(y_true, y_pred, weight = weight, reduce = False)
    #_loss = tf.reduce_sum(_loss, axis = -1)
    
    true_count = tf.reduce_sum(tf.cast(true_flag, y_pred.dtype))
    _loss = tf.reduce_sum(_loss) / tf.maximum(true_count, 1.)
    _loss = tf.where(tf.math.is_nan(_loss), missing_value, _loss)
    return _loss

def boxnet_loss(y_true, bbox_true, bbox_pred, loss = smooth_l1, missing_value = 0.):
    """
    y_true = targeted y_true #(batch_size, sampling_count, 1 or num_class)
    bbox_true = targeted true bbox #(batch_size, sampling_count, delta)
    bbox_pred = targeted pred bbox #(batch_size, sampling_count, delta)
    """
    y_true = tf.cond(tf.equal(tf.shape(y_true)[-1], 1), true_fn = lambda: y_true, false_fn = lambda: tf.expand_dims(tf.cast(tf.argmax(y_true, -1), y_true.dtype), axis = -1))
    y_true = tf.reshape(y_true, (-1,))
    bbox_true = tf.reshape(bbox_true, (-1, 4))
    bbox_pred = tf.reshape(bbox_pred, (-1, 4))
    
    true_indices = tf.where(0 < y_true)[:, 0]
    bbox_true = tf.gather(bbox_true, true_indices)
    bbox_pred = tf.gather(bbox_pred, true_indices)
    
    _loss = loss(bbox_true, bbox_pred, reduce = False)
    _loss = tf.reduce_sum(_loss, axis = -1)
    
    _loss = tf.reduce_mean(_loss)
    _loss = tf.where(tf.math.is_nan(_loss), missing_value, _loss)
    return _loss