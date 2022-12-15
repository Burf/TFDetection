import tensorflow as tf

def regularize(model, weight_decay = 1e-4, loss = tf.keras.regularizers.l2):
    weight_decay = weight_decay() if callable(weight_decay) else weight_decay
    reg_loss = []
    for w in model.trainable_weights:
        if "gamma" not in w.name and "beta" not in w.name:
            reg_loss.append(loss(weight_decay)(w) / tf.cast(tf.size(w), w.dtype))
    return reg_loss