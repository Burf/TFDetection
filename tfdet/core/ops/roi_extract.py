import functools

import tensorflow as tf

@tf.function
def roi2level(bbox, n_level, input_shape = (224, 224)):
    dtype = bbox.dtype
    x1, y1, x2, y2 = tf.split(bbox, 4, axis = -1)
    h = y2 - y1
    w = x2 - x1

    bbox_area = h * w
    image_area = tf.cast(input_shape[0] * input_shape[1], dtype)

    roi_level = tf.floor(tf.math.log((tf.sqrt(bbox_area)) / ((tf.cast(56., dtype) / tf.sqrt(image_area)) + tf.keras.backend.epsilon())) / tf.math.log(tf.cast(2., dtype)))
    roi_level = tf.cast(roi_level, tf.int8)
    roi_level = tf.clip_by_value(roi_level, 0, n_level - 1)
    roi_level = tf.squeeze(roi_level, axis = -1)
    return roi_level

class RoiAlign(tf.keras.layers.Layer):
    def __init__(self, pool_size = 7, method = "bilinear", scatter = False, divisor = 100000, **kwargs):
        super(RoiAlign, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.method = method
        self.divisor = divisor
        self.func = functools.partial(self.roi_align, pool_size = self.pool_size, method = self.method, divisor = self.divisor) if not scatter else functools.partial(self.roi_align_scatter, pool_size = self.pool_size, method = self.method)
        
    @staticmethod
    @tf.function
    def roi_align_scatter(feature, bbox_pred, image_shape = [1024, 1024], pool_size = 7, method = "bilinear"):
        if not isinstance(feature, list):
            feature = [feature]
        pool_size = [pool_size, pool_size] if not isinstance(pool_size, (tuple, list)) else pool_size

        bbox_shape = tf.shape(bbox_pred)
        batch_size = bbox_shape[0]
        num_proposals = bbox_shape[1]
        valid_flag = tf.reduce_any(tf.greater(bbox_pred, 0), axis = -1)
        negative_indices = tf.where(~valid_flag)

        roi_level = roi2level(bbox_pred, len(feature), image_shape)
        roi_level = tf.tensor_scatter_nd_update(roi_level, negative_indices, -tf.ones(tf.shape(negative_indices)[0], dtype = tf.int8))
        
        x1, y1, x2, y2 = tf.split(bbox_pred, 4, axis = -1)
        bbox_pred = tf.concat([y1, x1, y2, x2], axis = -1)

        roi = tf.zeros([batch_size, num_proposals, *pool_size, tf.keras.backend.int_shape(feature[0])[-1]], dtype = feature[0].dtype)
        for level, x in enumerate(feature):
            level_flag = tf.equal(roi_level, level)#tf.logical_and(, valid_flag)
            level_indices = tf.where(level_flag)
            bbox_indices = tf.stop_gradient(tf.cast(level_indices[..., 0], tf.int32))
            
            bbox = tf.stop_gradient(tf.gather_nd(bbox_pred, level_indices))

            out = tf.image.crop_and_resize(image = x, boxes = bbox, box_indices = bbox_indices, crop_size = pool_size, method = method)
            out = tf.cast(out, x.dtype)
            
            roi = tf.tensor_scatter_nd_update(roi, level_indices, out)
        return roi
        
    @staticmethod
    @tf.function
    def roi_align(feature, bbox_pred, image_shape = [1024, 1024], pool_size = 7, method = "bilinear", divisor = 100000):
        if not isinstance(feature, list):
            feature = [feature]
        pool_size = [pool_size, pool_size] if not isinstance(pool_size, (tuple, list)) else pool_size

        bbox_shape = tf.shape(bbox_pred)
        batch_size = bbox_shape[0]
        num_proposals = bbox_shape[1]

        roi_level = roi2level(bbox_pred, len(feature), image_shape)
        
        x1, y1, x2, y2 = tf.split(bbox_pred, 4, axis = -1)
        bbox_pred = tf.concat([y1, x1, y2, x2], axis = -1)

        indices = []
        roi = []
        for level, x in enumerate(feature):
            level_indices = tf.where(tf.equal(roi_level, level))
            level_indices = tf.cast(level_indices, tf.int32)
            indices.append(level_indices)
            
            bbox = tf.stop_gradient(tf.gather_nd(bbox_pred, level_indices))
            bbox_indices = tf.stop_gradient(level_indices[:, 0])
            
            out = tf.image.crop_and_resize(image = x, boxes = bbox, box_indices = bbox_indices, crop_size = pool_size, method = method)
            roi.append(out)
        roi = tf.concat(roi, axis = 0)
        
        #https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py
        indices = tf.concat(indices, axis = 0)
        ind_size = tf.shape(indices)[0]
        sort_indices = indices[:, 0] * divisor + indices[:, 1]
        sort_indices = tf.nn.top_k(sort_indices, k = ind_size).indices[::-1]
        new_indices = tf.gather(tf.range(ind_size), sort_indices)
        roi = tf.gather(roi, new_indices)

        roi = tf.reshape(roi, [batch_size, num_proposals, *pool_size, tf.shape(feature[0])[-1]])
        return roi

    def call(self, inputs, image_shape = [1024, 1024]):
        feature, proposals = inputs
        roi = self.func(feature, proposals, image_shape = image_shape)
        return roi
    
    def get_config(self):
        config = super(RoiAlign, self).get_config()
        config["pool_size"] = self.pool_size
        config["method"] = self.method
        config["divisor"] = self.divisor
        return config