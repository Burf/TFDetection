import tensorflow as tf

def random_sampler(true_indices, positive_indices, negative_indices, sampling_count = 256, positive_ratio = 0.5, return_count = False):
    if isinstance(sampling_count, int) and 0 < sampling_count:
        positive_count = tf.cast(sampling_count * positive_ratio, tf.int32)
        indices = tf.range(tf.shape(positive_indices)[0])
        indices = tf.random.shuffle(indices)[:positive_count]
        positive_indices = tf.gather(positive_indices, indices)
        true_indices = tf.gather(true_indices, indices)
        positive_count = tf.cast(tf.shape(positive_indices)[0], tf.float32)
        negative_count = tf.cast(1 / positive_ratio * positive_count - positive_count, tf.int32)
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
    if return_count:
        return true_indices, positive_indices, negative_indices, sampling_count
    else:
        return true_indices, positive_indices, negative_indices