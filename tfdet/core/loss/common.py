import tensorflow as tf

def regularize(model, weight_decay = 1e-4, loss = tf.keras.regularizers.l2):
    weight_decay = weight_decay() if callable(weight_decay) else weight_decay
    reg_loss = []
    for w in model.trainable_weights:
        if "gamma" not in w.name and "beta" not in w.name:
            reg_loss.append(loss(weight_decay)(w) / tf.cast(tf.size(w), w.dtype))
    return reg_loss
    
def smooth_l1(y_true, y_pred, sigma = 1):
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, (1.0 / sigma ** 2)), diff.dtype)
    loss = (less_than_one * (0.5 * sigma ** 2) * diff ** 2) + (1 - less_than_one) * (diff - (0.5 / sigma ** 2))
    return loss