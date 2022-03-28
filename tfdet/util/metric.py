import tensorflow as tf

from ..core.util.tf import map_fn

def mean_average_precision(y_true, bbox_true, y_pred, bbox_pred, threshold = 0.5, r = 11, interpolate = True, batch_size = 10, reduce = True, return_precision_n_recall = False):
    """
    y_true = label #(batch_size, padded_num_true, 1)
    bbox_true = [[x1, y1, x2, y2], ...] #(batch_size, padded_num_true, bbox)
    y_pred = classifier logit   #(batch_size, num_proposals, num_class)
    bbox_pred = classifier regress #(batch_size, num_proposals, delta)
    """
    n_batch = tf.shape(y_true)[0]
    n_class = tf.keras.backend.int_shape(y_pred)[-1]
    cls_indices = tf.range(n_class)
    r_threshold = tf.linspace(0., 1., r)
    
    def metric(tp, fp, fn, batch_index):
        _y_true, _bbox_true, _y_pred, _bbox_pred = [tensor[batch_index] for tensor in [y_true, bbox_true, y_pred, bbox_pred]]
        valid_true_indices = tf.where(tf.reduce_max(tf.cast(0 < _bbox_true, tf.int32), axis = -1))
        _y_true = tf.cast(tf.gather_nd(_y_true, valid_true_indices), tf.int32)
        _bbox_true = tf.gather_nd(_bbox_true, valid_true_indices)
        valid_pred_indices = tf.where(tf.reduce_max(tf.cast(0 < _bbox_pred, tf.int32), axis = -1))
        _y_pred = tf.gather_nd(_y_pred, valid_pred_indices)
        _bbox_pred = tf.gather_nd(_bbox_pred, valid_pred_indices)
        
        label = tf.argmax(_y_pred, axis = -1, output_type = tf.int32)
        indices = tf.stack([tf.range(tf.shape(_y_pred)[0]), label], axis = -1)
        score = tf.gather_nd(_y_pred, indices)
        overlaps = threshold <= overlap_bbox(_bbox_true, _bbox_pred)
        
        def cls_body(tp, fp, fn, cls_id):
            true_indices = tf.where(tf.equal(_y_true, cls_id))[..., 0]
            true_count = tf.cast(tf.shape(true_indices)[0], tf.float32)
            pred_flag = tf.equal(label, cls_id)
            pred_count = tf.reduce_sum(tf.cast(pred_flag, tf.float32))
            
            mask = None
            if pred_count == 0:
                if true_count == 0:
                    return tp, fp, fn
                else:
                    cls_indices = tf.stack([tf.ones(r, dtype = tf.int32) * cls_id, tf.range(11)], axis = -1)
                    return tp, fp, tf.tensor_scatter_nd_update(fn, cls_indices, fn[cls_id] + true_count)
            else:
                if true_count != 0:
                    mask = tf.gather(overlaps, true_indices, axis = 1)
                
                def r_body(cls_tp, cls_fp, cls_fn, r_index):
                    _pred_flag = tf.logical_and(pred_flag, tf.greater_equal(score, r_threshold[r_index]))
                    _pred_count = tf.reduce_sum(tf.cast(_pred_flag, tf.float32))
                    if true_count == 0:
                        return cls_tp, tf.tensor_scatter_nd_update(cls_fp, [[r_index]], [cls_fp[r_index] + _pred_count]), cls_fn
                    else:
                        _mask = tf.gather_nd(mask, tf.where(_pred_flag))
                        tp_count = tf.reduce_sum(tf.cast(tf.reduce_any(_mask, axis = 0), tf.float32))
                        return tf.tensor_scatter_nd_update(cls_tp, [[r_index]], [cls_tp[r_index] + tp_count]),\
                               tf.tensor_scatter_nd_update(cls_fp, [[r_index]], [cls_fp[r_index] + (_pred_count - tp_count)]),\
                               tf.tensor_scatter_nd_update(cls_fn, [[r_index]], [cls_fn[r_index] + (true_count - tp_count)])

                cls_tp, cls_fp, cls_fn = tf.while_loop(lambda index, cls_tp, cls_fp, cls_fn: index < r,
                                                       lambda index, cls_tp, cls_fp, cls_fn: (index + 1, *r_body(cls_tp, cls_fp, cls_fn, index)),
                                                       (0, tp[cls_id], fp[cls_id], fn[cls_id]),
                                                       parallel_iterations = 1)[1:]
                
                cls_indices = tf.stack([tf.ones(r, dtype = tf.int32) * cls_id, tf.range(11)], axis = -1)
                tp = tf.tensor_scatter_nd_update(tp, cls_indices, cls_tp)
                fp = tf.tensor_scatter_nd_update(fp, cls_indices, cls_fp)
                fn = tf.tensor_scatter_nd_update(fn, cls_indices, cls_fn)
                return tp, fp, fn
        
        tp, fp, fn = tf.while_loop(lambda index, tp, fp, fn: index < n_class,
                                   lambda index, tp, fp, fn: (index + 1, *cls_body(tp, fp, fn, cls_indices[index])),
                                   (0, tp, fp, fn),
                                   parallel_iterations = 1)[1:]
        return tp, fp, fn
    
    tp = tf.zeros((n_class, r), dtype = tf.float32)
    fp = tf.zeros((n_class, r), dtype = tf.float32)
    fn = tf.zeros((n_class, r), dtype = tf.float32)
    tp, fp, fn = tf.while_loop(lambda index, tp, fp, fn: index < n_batch,
                               lambda index, tp, fp, fn: (index + 1, *metric(tp, fp, fn, index)),
                               (0, tp, fp, fn),
                               parallel_iterations = batch_size)[1:]
    
    tp_fp = tp + fp
    tp_fn = tp + fn
    precision = tf.where(tf.equal(tp_fp, 0.), tf.where(tf.equal(tp_fn, 0.), 1., 0.), tp / tp_fp)
    recall = tf.where(tf.equal(tp_fn, 0.), 1., tp / tp_fn)
    if interpolate:
        def interpolation(precision, r_index):
            new_precision = tf.reduce_max(precision[..., :r_index + 1], axis = 1)
            r_indices = tf.stack([cls_indices, tf.ones(n_class, dtype = tf.int32) * r_index], axis = -1)
            return tf.tensor_scatter_nd_update(precision, r_indices, new_precision)
        
        precision = tf.while_loop(lambda index, precision: index < r,
                                  lambda index, precision: (index + 1, interpolation(precision, index)),
                                  (1, precision),
                                  parallel_iterations = batch_size)[1]
    
    if return_precision_n_recall:
        return precision, recall
    
    average_precision = tf.reduce_sum(precision[..., ::-1] * (recall[..., ::-1] - tf.concat([tf.zeros((n_class, 1), dtype = tf.float32), recall[..., 1:][..., ::-1]], axis = -1)), axis = -1)
    if reduce:
        average_precision = tf.reduce_mean(average_precision)
    return average_precision