import tensorflow as tf

from tfdet.core.loss import categorical_cross_entropy, regularize as regularize_loss
from .loss import ResizeMaskLoss

def train_model(input, mask_pred, aux_mask_pred = None,
                loss = categorical_cross_entropy,
                regularize = True, weight_decay = 1e-4,
                class_weight = None, method = "bilinear", aux_weight = 0.4, 
                missing_value = 0.):
    mask_pred = [mask_pred] if not isinstance(mask_pred, (tuple, list)) else mask_pred
    if aux_mask_pred is None and 1 < len(mask_pred):
        aux_mask_pred = mask_pred[1]
    mask_pred = mask_pred[0]
    
    mask_true = tf.keras.layers.Input(shape = (None, None, None), dtype = tf.uint16, name = "mask_true")
    _loss = ResizeMaskLoss(loss = loss, weight = class_weight, method = method, missing_value = missing_value, dtype = tf.float32, name = "mask_loss")(mask_true, mask_pred)
    aux_loss = None
    if aux_mask_pred is not None:
        aux_loss = ResizeMaskLoss(loss = loss, weight = class_weight, method = method, missing_value = missing_value, dtype = tf.float32, name = "aux_mask_loss")(mask_true, aux_mask_pred)
        aux_loss *= aux_weight
    model = tf.keras.Model([input, mask_true], mask_pred)
    
    _loss = tf.expand_dims(_loss, axis = -1)
    if aux_loss is not None:
        model.add_metric(_loss, name = "loss_mask", aggregation = "mean")
        aux_loss = tf.expand_dims(aux_loss, axis = -1)
        model.add_metric(aux_loss, name = "loss_aux_mask", aggregation = "mean")
        _loss = tf.reduce_sum([_loss, aux_loss], axis = 0)
    model.add_metric(_loss, name = "loss", aggregation = "mean")
    model.add_loss(_loss)
    
    if regularize:
        model.add_loss(lambda: tf.cast(tf.reduce_sum(regularize_loss(model, weight_decay), keepdims = True), tf.float32))
    return model