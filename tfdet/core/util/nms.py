import tensorflow as tf

def pad_nms(proposal, score, proposal_count, iou_threshold, score_threshold = float('-inf'), soft_nms = True):
    soft_nms_sigma = soft_nms
    if not isinstance(soft_nms, float):
        soft_nms_sigma = 0.5 if soft_nms else 0.
    indices = tf.image.non_max_suppression_with_scores(proposal, score, max_output_size = tf.minimum(proposal_count, tf.shape(proposal)[0]), iou_threshold = iou_threshold, score_threshold = score_threshold, soft_nms_sigma = soft_nms_sigma, name = "non_maximum_suppression")[0]
    proposal = tf.gather(proposal, indices)
    pad_size = proposal_count - tf.shape(proposal)[0]
    proposal = tf.pad(proposal, [[0, pad_size], [0, 0]])
    return proposal