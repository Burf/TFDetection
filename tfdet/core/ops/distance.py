import tensorflow as tf

def mahalanobis(u, v, VI):
    delta = u - v
    out = tf.sqrt(tf.matmul(tf.matmul(tf.expand_dims(delta, axis = -2), VI), tf.expand_dims(delta, axis = -1)))
    return tf.squeeze(out, axis = -1)
    
def euclidean(u, v):
    return tf.sqrt(tf.reduce_sum(tf.square(u - v), axis = -1))

def euclidean_matrix(u, v):
    u_norm = tf.reduce_sum(tf.square(u), axis = -1, keepdims = True)
    v_norm = tf.reduce_sum(tf.square(v), axis = -1, keepdims = True)
    dist = tf.sqrt(tf.maximum(tf.add(tf.transpose(v_norm) - 2 * tf.matmul(u, v, transpose_b = True), u_norm), 0))
    return dist
