import tensorflow as tf

def regularize(model, weight_decay = 1e-4, loss = tf.keras.regularizers.l2):
    weight_decay = weight_decay() if callable(weight_decay) else weight_decay
    reg_loss = []
    for w in model.trainable_weights:
        if "gamma" not in w.name and "beta" not in w.name:
            l = loss(weight_decay)(w)
            reg_loss.append(l / tf.cast(tf.size(w), l.dtype))
    return reg_loss