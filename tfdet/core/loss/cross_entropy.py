import tensorflow as tf

def binary_cross_entropy(y_true, y_pred, focal = False, alpha = .25, gamma = 1.5, weight = None, reduce = tf.reduce_mean):
    n_true_class = tf.shape(y_true)[-1]
    n_pred_class = tf.shape(y_pred)[-1]
    
    y_true = tf.cast(tf.cond(tf.logical_and(tf.equal(n_true_class, 1), tf.not_equal(n_pred_class, 1)), true_fn = lambda: tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), n_pred_class)[..., 0, :], y_true.dtype), false_fn = lambda: y_true), y_pred.dtype)
    #y_pred = y_pred / (tf.reduce_sum(y_pred, axis = -1, keepdims = True) + tf.keras.backend.epsilon())
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    
    loss = -tf.stack([(1 - y_true) * tf.math.log(1 - y_pred), y_true * tf.math.log(y_pred)], axis = -1)
    if focal:
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_factor = tf.where(0.5 < y_true, alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(0.5 < y_true, 1 - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma
        loss = tf.expand_dims(focal_weight, axis = -1) * loss
    loss = tf.reduce_sum(loss, axis = -1)
    if weight is not None:
        loss *= weight
    if reduce:
        axis = tf.range(tf.rank(y_true))[1:]
        loss = reduce(loss, axis = axis)
        loss = tf.reduce_mean(loss)
    return loss

def categorical_cross_entropy(y_true, y_pred, focal = False, alpha = 1., gamma = 2., weight = None, reduce = tf.reduce_mean):
    n_true_class = tf.shape(y_true)[-1]
    n_pred_class = tf.shape(y_pred)[-1]
    
    y_true = tf.cast(tf.cond(tf.logical_and(tf.equal(n_true_class, 1), tf.not_equal(n_pred_class, 1)), true_fn = lambda: tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), n_pred_class)[..., 0, :], y_true.dtype), false_fn = lambda: y_true), y_pred.dtype)
    #y_pred = y_pred / (tf.reduce_sum(y_pred, axis = -1, keepdims = True) + tf.keras.backend.epsilon())
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    
    loss = -y_true * tf.math.log(y_pred)
    if focal:
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * loss
    if weight is not None:
        loss *= weight
    if reduce:
        axis = tf.range(tf.rank(y_true))[1:]
        loss = reduce(loss, axis = axis)
        loss = tf.reduce_mean(loss)
    return loss

def focal_binary_cross_entropy(y_true, y_pred, alpha = .25, gamma = 1.5, weight = None, reduce = tf.reduce_mean):
    return binary_cross_entropy(y_true, y_pred, focal = True, alpha = alpha, gamma = gamma, weight = weight, reduce = reduce)

def focal_categorical_cross_entropy(y_true, y_pred, alpha = 1., gamma = 2., weight = None, reduce = tf.reduce_mean):
    return categorical_cross_entropy(y_true, y_pred, focal = True, alpha = alpha, gamma = gamma, weight = weight, reduce = reduce)