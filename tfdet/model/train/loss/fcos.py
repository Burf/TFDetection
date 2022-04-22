import tensorflow as tf

from .retina import classnet_accuracy, classnet_loss, boxnet_loss

def centernessnet_loss(y_true, centerness_true, centerness_pred, missing_value = 0.):
    """
    y_true = targeted y_true #(batch_size, sampling_count, 1 or num_class)
    centerness_true = targeted centerness_true #(batch_size, sampling_count, 1)
    centerness_pred = targeted centerenss_pred #(batch_size, sampling_count, 1)
    """
    y_true = tf.cond(tf.equal(tf.shape(y_true)[-1], 1), true_fn = lambda: y_true, false_fn = lambda: tf.expand_dims(tf.cast(tf.argmax(y_true, -1), y_true.dtype), axis = -1))
    y_true = tf.reshape(y_true, (-1,))
    centerness_true = tf.reshape(centerness_true, (-1, 1))
    centerness_pred = tf.reshape(centerness_pred, (-1, 1))
    
    true_indices = tf.where(0 < y_true)
    centerness_true = tf.gather_nd(centerness_true, true_indices)
    centerness_pred = tf.gather_nd(centerness_pred, true_indices)
    
    loss = tf.keras.losses.binary_crossentropy(centerness_true, centerness_pred)
    loss = tf.reduce_mean(loss)
    loss = tf.where(tf.math.is_nan(loss), missing_value, loss)
    return loss