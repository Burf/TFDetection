import tensorflow as tf

from tfdet.core.loss import smooth_l1
    
def classnet_accuracy(y_true, y_pred, missing_value = 0.):
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
    
    dtype = y_pred.dtype
    y_true = tf.cond(tf.equal(n_true_class, 1), true_fn = lambda: y_true[..., 0], false_fn = lambda: tf.cast(tf.argmax(y_true, axis = -1), y_true.dtype))
    y_pred = tf.argmax(y_pred, axis = -1)
    y_true = tf.cast(y_true, dtype)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), dtype))
    accuracy = tf.where(tf.math.is_nan(accuracy), missing_value, accuracy)
    return accuracy

def classnet_loss(y_true, y_pred, focal = True, alpha = .25, gamma = 1.5, weight = None, missing_value = 0.):
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
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    
    loss = -tf.stack([(1 - y_true) * tf.math.log(1 - y_pred), y_true * tf.math.log(y_pred)], axis = -1)
    if focal:
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_factor = tf.where(0.5 < y_true, alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(0.5 < y_true, 1 - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma
        loss = tf.expand_dims(focal_weight, axis = -1) * loss
    if weight is not None:
        loss *= weight
    loss = tf.reduce_sum(loss, axis = -1)
    loss = tf.reduce_mean(loss, axis = -1)
    
    loss = tf.reduce_mean(loss)
    loss = tf.where(tf.math.is_nan(loss), missing_value, loss)
    return loss

def boxnet_loss(y_true, bbox_true, bbox_pred, sigma = 3, missing_value = 0.):
    """
    y_true = targeted y_true #(batch_size, sampling_count, 1 or num_class)
    bbox_true = targeted true bbox #(batch_size, sampling_count, delta)
    bbox_pred = targeted pred bbox #(batch_size, sampling_count, delta)
    """
    y_true = tf.cond(tf.equal(tf.shape(y_true)[-1], 1), true_fn = lambda: y_true, false_fn = lambda: tf.expand_dims(tf.cast(tf.argmax(y_true, -1), y_true.dtype), axis = -1))
    y_true = tf.reshape(y_true, (-1,))
    bbox_true = tf.reshape(bbox_true, (-1, 4))
    bbox_pred = tf.reshape(bbox_pred, (-1, 4))
    
    true_indices = tf.where(0 < y_true)
    bbox_true = tf.gather_nd(bbox_true, true_indices)
    bbox_pred = tf.gather_nd(bbox_pred, true_indices)
    
    loss = smooth_l1(bbox_true, bbox_pred, sigma)
    
    loss = tf.reduce_mean(loss)
    loss = tf.where(tf.math.is_nan(loss), missing_value, loss)
    return loss