import tensorflow as tf

def regularize_loss(model, weight_decay = 1e-4, loss = tf.keras.regularizers.l2):
    reg_loss = []
    for w in model.trainable_weights:
        if "gamma" not in w.name and "beta" not in w.name:
            reg_loss.append(loss(weight_decay)(w) / tf.cast(tf.size(w), tf.float32))
    return reg_loss