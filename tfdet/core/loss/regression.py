import tensorflow as tf
    
def smooth_l1(y_true, y_pred, sigma = 3, reduce = tf.reduce_mean):
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, (1.0 / sigma ** 2)), diff.dtype)
    loss = (less_than_one * (0.5 * sigma ** 2) * diff ** 2) + (1 - less_than_one) * (diff - (0.5 / sigma ** 2))
    if reduce:
        loss = reduce(loss, axis = -1)
        loss = tf.reduce_mean(loss)
    return loss