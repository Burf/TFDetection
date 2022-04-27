import tensorflow as tf

def overlap_bbox(bbox_true, bbox_pred, mode = "normal"):
    """
    bbox_true = [[x1, y1, x2, y2], ...] #(N, bbox)
    bbox_pred = [[x1, y1, x2, y2], ...] #(M, bbox)
    
    overlaps = pred & true iou matrix #(M, N)
    """
    if mode not in ("normal", "foreground", "general", "complete", "distance"):
        raise ValueError("unknown mode '{0}'".format(mode))
    
    true_count = tf.shape(bbox_true)[0]
    pred_count = tf.shape(bbox_pred)[0]
        
    bbox_true = tf.reshape(tf.tile(tf.expand_dims(bbox_true, 0), [1, 1, pred_count]), [-1, 4])
    bbox_pred = tf.tile(bbox_pred, [true_count, 1])
    
    tx1, ty1, tx2, ty2 = tf.split(bbox_true, 4, axis = -1)
    px1, py1, px2, py2 = tf.split(bbox_pred, 4, axis = -1)
    
    area_true = (tx2 - tx1) * (ty2 - ty1)
    area_pred = (px2 - px1) * (py2 - py1)
    
    x1 = tf.maximum(tx1, px1)
    y1 = tf.maximum(ty1, py1)
    x2 = tf.minimum(tx2, px2)
    y2 = tf.minimum(ty2, py2)
    
    inter = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    if mode != "foreground":
        union = area_true + area_pred - inter
    else:
        union = area_true
    iou = tf.clip_by_value((inter + tf.keras.backend.epsilon()) / (union + tf.keras.backend.epsilon()), 0, 1)
    if mode in ["general", "complete", "distance"]:
        gx1 = tf.minimum(tx1, px1)
        gy1 = tf.minimum(ty1, py1)
        gx2 = tf.maximum(tx2, px2)
        gy2 = tf.maximum(ty2, py2)
        
        if mode == "general":
            general_inter = tf.maximum(gx2 - gx1, 0) * tf.maximum(gy2 - gy1, 0)
            iou = giou = tf.clip_by_value(iou - (general_inter - union + tf.keras.backend.epsilon()) / (general_inter + tf.keras.backend.epsilon()), 0, 1)
        else:
            tw = tf.maximum(tx2 - tx1, 0)
            th = tf.maximum(ty2 - ty1, 0)
            pw = tf.maximum(px2 - px1, 0)
            ph = tf.maximum(py2 - py1, 0)
            ctx = tx1 + 0.5 * tw
            cty = ty1 + 0.5 * th
            cpx = px1 + 0.5 * pw
            cpy = py1 + 0.5 * ph
            c = (gx2 - gx1) ** 2 + (gy2 - gy1) ** 2
            rho = (ctx - cpx) ** 2 + (cty - cpy) ** 2
            
            diou = tf.clip_by_value(iou - (rho + tf.keras.backend.epsilon()) / (c + tf.keras.backend.epsilon()), 0, 1)
            if mode == "distance": 
                iou = diou
            else: #complete
                v = ((tf.math.atan((pw + tf.keras.backend.epsilon()) / (ph + tf.keras.backend.epsilon()))
                    - tf.math.atan((tw + tf.keras.backend.epsilon()) / (th + tf.keras.backend.epsilon()))) * 2 / np.pi) ** 2
                alpha = (v + tf.keras.backend.epsilon()) / (1 - iou + v + tf.keras.backend.epsilon())
                iou = ciou = tf.clip_by_value(diou - alpha * v, 0, 1)
    overlaps = tf.reshape(iou, [true_count, pred_count])
    return overlaps

def overlap_fcos(bbox_true, points, regress_range):
    true_count = tf.shape(bbox_true)[0]
    pred_count = tf.shape(points)[0]

    bbox_true = tf.reshape(tf.tile(tf.expand_dims(bbox_true, 0), [1, 1, pred_count]), [-1, 4])
    points = tf.tile(points, [true_count, 1])
    regress_range = tf.tile(regress_range, [true_count, 1])
    
    x1, y1, x2, y2 = tf.split(bbox_true, 4, axis = -1)
    px, py = tf.split(points, 2, axis = -1)
    area = tf.reshape((x2 - x1) * (y2 - y1), [true_count, pred_count])
    offset = tf.concat([px - x1, py - y1, x2 - px, y2 - py], axis = -1) #left, top, right, bottom
    offset = tf.reshape(offset, [true_count, pred_count, 4])
    regress_range = tf.reshape(regress_range, [true_count, pred_count, 2])
    min_offset = tf.reduce_min(offset, axis = -1)
    max_offset = tf.reduce_max(offset, axis = -1)
    
    overlap_flag = tf.greater(min_offset, 0)
    range_flag = tf.logical_and(tf.greater(max_offset, regress_range[..., 0]), tf.less_equal(max_offset, regress_range[..., 1]))
    pad_area = tf.where(tf.logical_and(overlap_flag, range_flag), area, tf.reduce_max(area) + 1)
    min_flag = tf.equal(area, tf.reduce_min(pad_area, axis = 0, keepdims = True))
    overlaps = tf.where(min_flag, area, 0)
    return overlaps