import tensorflow as tf

from tfdet.core.bbox import delta2bbox
from tfdet.core.util.nms import pad_nms
from tfdet.core.util.tf import map_fn

class RegionProposalNetwork(tf.keras.layers.Layer):
    def __init__(self, n_anchor, share = True, n_feature = 512, use_bias = False, activation = tf.keras.activations.relu, **kwargs):
        super(RegionProposalNetwork, self).__init__(**kwargs)   
        self.n_anchor = n_anchor
        self.share = share
        self.n_feature = n_feature
        self.use_bias = use_bias
        self.activation = activation

        self.score_reshape = tf.keras.layers.Reshape((-1, 1), name = "score")
        self.delta = tf.keras.layers.Reshape((-1, 4), name = "delta")

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        if self.share:
            self.feature = [tf.keras.layers.Conv2D(self.n_feature, (3, 3), padding = "same", use_bias = False, activation = self.activation, name = "shared_feature_conv")] * len(input_shape)
            self.score = [tf.keras.layers.Conv2D(self.n_anchor, (1, 1), use_bias = self.use_bias, activation = tf.keras.activations.sigmoid, name = "shared_score_conv")] * len(input_shape)
            self.regress = [tf.keras.layers.Conv2D(self.n_anchor * 4, (1, 1), use_bias = self.use_bias, activation = tf.keras.activations.linear, name = "shared_regress_conv")] * len(input_shape)
        else:
            self.feature = [tf.keras.layers.Conv2D(self.n_feature, (3, 3), padding = "same", use_bias = False, activation = self.activation, name = "feature_conv{0}".format(index + 1)) for index in range(len(input_shape))]
            self.score = [tf.keras.layers.Conv2D(self.n_anchor, (1, 1), use_bias = self.use_bias, activation = tf.keras.activations.sigmoid, name = "score_conv{0}".format(index + 1)) for index in range(len(input_shape))]
            self.regress = [tf.keras.layers.Conv2D(self.n_anchor * 4, (1, 1), use_bias = self.use_bias, activation = tf.keras.activations.linear, name = "regress_conv{0}".format(index + 1)) for index in range(len(input_shape))]
        
        if 1 < len(input_shape):
            self.score_concat = tf.keras.layers.Concatenate(axis = -2, name = "score_concat")
            self.delta_concat = tf.keras.layers.Concatenate(axis = -2, name = "delta_concat")

    def call(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        out = []
        for i, x in enumerate(inputs):
            feature = self.feature[i](x)
            score = self.score[i](feature)
            regress = self.regress[i](feature)
            score = self.score_reshape(score)
            delta = self.delta(regress)
            out.append([score, delta])
        out = list(zip(*out))
        if len(out[0]) == 1:
            out = [o[0] for o in out]
        else:
            out[0] = self.score_concat(out[0])
            out[1] = self.delta_concat(out[1])
        return out
    
    def get_config(self):
        config = super(RegionProposalNetwork, self).get_config()
        config["n_anchor"] = self.n_anchor
        config["share"] = self.share
        config["n_feature"] = self.n_feature
        config["use_bias"] = self.use_bias
        config["activation"] = self.activation
        return config
    
class Rpn2Proposal(tf.keras.layers.Layer):
    def __init__(self, proposal_count = 1000, iou_threshold = 0.7, soft_nms = True, valid = True, performance_count = 5000, batch_size = 1, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, **kwargs):
        super(Rpn2Proposal, self).__init__(**kwargs)   
        self.proposal_count = proposal_count
        self.iou_threshold = iou_threshold
        self.soft_nms = soft_nms
        self.valid = valid
        self.performance_count = performance_count
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.clip_ratio = clip_ratio

    def call(self, inputs, anchors):
        rpn_score, rpn_regress = inputs
        rpn_score = tf.squeeze(rpn_score, axis = -1)
        
        # Proposal valid anchors
        if self.valid:
            valid_flags = tf.logical_and(tf.less_equal(anchors[..., 2], 1),
                                         tf.logical_and(tf.less_equal(anchors[..., 3], 1),
                                                        tf.logical_and(tf.greater_equal(anchors[..., 0], 0),
                                                                       tf.greater_equal(anchors[..., 1], 0))))
            #valid_indices = tf.range(tf.shape(rpn_score)[1])[valid_flags]
            valid_indices = tf.where(valid_flags)[:, 0]
            rpn_score = tf.gather(rpn_score, valid_indices, axis = 1)
            rpn_regress = tf.gather(rpn_regress, valid_indices, axis = 1)
            anchors = tf.gather(anchors, valid_indices)
        
        # Transform delta to bbox
        proposals = delta2bbox(anchors, rpn_regress, self.mean, self.std, self.clip_ratio)
        
        # Proposal sorted anchors by performance_count
        performance_count = tf.minimum(self.performance_count, tf.shape(proposals)[1])
        top_indices = tf.nn.top_k(rpn_score, performance_count, sorted = True).indices
        rpn_score = tf.gather(rpn_score, top_indices, batch_dims = 1)
        proposals = tf.gather(proposals, top_indices, batch_dims = 1)

        # Clipping to valid area
        proposals = tf.clip_by_value(proposals, 0, 1)

        # Transform
        x1, y1, x2, y2 = tf.split(proposals, 4, axis = -1)
        proposals = tf.concat([y1, x1, y2, x2], axis = -1)

        # NMS
        proposals = map_fn(pad_nms, proposals, rpn_score, dtype = tf.float32, batch_size = self.batch_size, 
                           proposal_count = self.proposal_count, iou_threshold = self.iou_threshold, soft_nms = self.soft_nms)
        proposals = tf.reshape(proposals, [-1, self.proposal_count, 4])

        # Transform
        y1, x1, y2, x2 = tf.split(proposals, 4, axis = -1)
        proposals = tf.concat([x1, y1, x2, y2], axis = -1)
        proposals = tf.stop_gradient(proposals)
        return proposals
    
    def get_config(self):
        config = super(Rpn2Proposal, self).get_config()
        config["proposal_count"] = self.proposal_count
        config["iou_threshold"] = self.iou_threshold
        config["soft_nms"] = self.soft_nms
        config["valid"] = self.valid
        config["performance_count"] = self.performance_count
        config["batch_size"] = self.batch_size
        config["mean"] = self.mean
        config["std"] = self.std
        config["clip_ratio"] = self.clip_ratio
        return config

class RoiAlign(tf.keras.layers.Layer):
    def __init__(self, pool_size = 7, method = "bilinear", **kwargs):
        super(RoiAlign, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.method = method

    def call(self, inputs, image_shape = [1024, 1024]):
        feature, proposals = inputs
        if not isinstance(feature, list):
            feature = [feature]
        pool_size = self.pool_size
        if isinstance(pool_size, int):
            pool_size = [pool_size, pool_size]
        
        roi_level = roi2level(proposals, len(feature), image_shape)
        x1, y1, x2, y2 = tf.split(proposals, 4, axis = -1)
        proposals = tf.concat([y1, x1, y2, x2], axis = -1)
        align = tf.keras.layers.Lambda(lambda args: tf.image.crop_and_resize(image = args[0], boxes = args[1], box_indices = args[2], crop_size = pool_size, method = self.method))
    
        indices = []
        result = []
        for level, x in enumerate(feature):
            level_indices = tf.where(tf.equal(roi_level, level))
            bbox = tf.gather_nd(proposals, level_indices)

            bbox = tf.stop_gradient(bbox)
            bbox_indices = tf.stop_gradient(tf.cast(level_indices[:, 0], tf.int32))
            out = align([x, bbox, bbox_indices])

            indices.append(level_indices)
            result.append(out)

        indices = tf.concat(indices, axis = 0)
        result = tf.concat(result, axis = 0)
          
        # rearange
        sort_range = tf.expand_dims(tf.range(tf.shape(indices)[0]), axis = 1)
        indices = tf.concat([tf.cast(indices, tf.int32), sort_range], axis = 1)

        sort_indices = indices[:, 0] * 100000 + indices[:, 1]
        sorted_indices = tf.nn.top_k(sort_indices, k = tf.shape(indices)[0]).indices[::-1]
        indices = tf.gather(indices[:, 2], sorted_indices)
        result = tf.gather(result, indices)
        
        # reshape
        shape = tf.concat([tf.shape(proposals)[:2], tf.shape(result)[1:]], axis = 0)
        result = tf.reshape(result, shape)
        return result
    
    def get_config(self):
        config = super(RoiAlign, self).get_config()
        config["pool_size"] = self.pool_size
        config["method"] = self.method
        return config

class RoiClassifier(tf.keras.layers.Layer):
    def __init__(self, n_class = 21, n_feature = 1024, activation = tf.keras.activations.relu, **kwargs):
        super(RoiClassifier, self).__init__(**kwargs)   
        self.n_class = n_class
        self.n_feature = n_feature
        self.activation = activation

        self.bn1 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(axis = -1), name = "pooling_bn")
        self.act1 = tf.keras.layers.Activation(activation, name = "pooling_act")
        self.conv2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(n_feature, 1, use_bias = True, kernel_initializer = "he_normal", bias_initializer = "zeros"), name = "feature_conv")
        self.bn2 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(axis = -1), name = "feature_bn")
        self.act2 = tf.keras.layers.Activation(activation, name = "feature_act")
        self.feature = tf.keras.layers.Reshape([-1, n_feature], name = "shared_feature")
        self.logits = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_class, activation = tf.keras.activations.softmax), name = "logits")
        self.regress = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_class * 4, activation = tf.keras.activations.linear), name = "regress")
        self.delta = tf.keras.layers.Reshape([-1, n_class, 4], name = "delta")

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(self.n_feature, input_shape[-3:-1], padding = "valid", use_bias = True, kernel_initializer = "he_normal", bias_initializer = "zeros"), name = "pooling_conv")

    def call(self, inputs):
        out = inputs
        for layer in [self.conv1, self.bn1, self.act1, 
                      self.conv2, self.bn2, self.act2]:
            out = layer(out)
        out = self.feature(out)
        logits = self.logits(out)
        regress = self.regress(out)
        regress = self.delta(regress)
        return logits, regress
    
    def get_config(self):
        config = super(RoiClassifier, self).get_config()
        config["n_class"] = self.n_class
        config["n_feature"] = self.n_feature
        config["activation"] = self.activation
        return config

class RoiMask(tf.keras.layers.Layer):
    def __init__(self, n_class = 21, n_feature = 256, n_depth = 4, activation = tf.keras.activations.relu, **kwargs):
        super(RoiMask, self).__init__(**kwargs)   
        self.n_class = n_class
        self.n_feature = n_feature
        self.n_depth = n_depth
        self.activation = activation
        
        self.layers = []        
        for index in range(n_depth):
            self.layers.append(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(n_feature, 3, use_bias = True, padding = "same", kernel_initializer = "he_normal", bias_initializer = "zeros"), name = "feature_conv{0}".format(index + 1)))
            self.layers.append(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(axis = -1), name = "feature_bn{0}".format(index + 1)))
            self.layers.append(tf.keras.layers.Activation(activation, name = "feature_act{0}".format(index + 1)))
        self.deconv = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(256, (2, 2), strides = 2, activation = activation), name = "deconv")
        self.mask = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(n_class, (1, 1), activation = tf.keras.activations.sigmoid), name = "mask")
        
    def build(self, input_shape):
        if isinstance(input_shape, list):
            self.resample = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(input_shape[0][-1], 1, use_bias = True, kernel_initializer = "he_normal", bias_initializer = "zeros"), name = "resample_conv")

    def call(self, inputs, feature = False):
        out = inputs
        if isinstance(inputs, list):
            out, residual = inputs
            if residual is not None:
                out = out + self.resample(residual)
        for layer in self.layers:
            out = layer(out)
        deconv = self.deconv(out)
        mask = self.mask(deconv)
        if feature:
            return mask, out  
        else:
            return mask
    
    def get_config(self):
        config = super(RoiMask, self).get_config()
        config["n_class"] = self.n_class
        config["n_feature"] = self.n_feature
        config["n_depth"] = self.n_depth
        config["activation"] = self.activation
        return config

def classifier2proposal(cls_logit, cls_regress, proposal, valid = True, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000):
    if valid:
        sampling_count = tf.shape(proposal)[0]
        valid_indices = tf.where(tf.reduce_max(tf.cast(0 < proposal, tf.int32), axis = -1))
        cls_logit = tf.gather_nd(cls_logit, valid_indices)
        cls_regress = tf.gather_nd(cls_regress, valid_indices)
        proposal = tf.gather_nd(proposal, valid_indices)

    logit_indices = tf.argmax(cls_logit, axis = -1, output_type = tf.int32)
    indices = tf.stack([tf.range(tf.shape(cls_logit)[0]), logit_indices], axis = -1)
    delta = tf.gather_nd(cls_regress, indices)

    proposal = delta2bbox(proposal, delta, mean, std, clip_ratio) # Transform delta to bbox
    proposal = tf.clip_by_value(proposal, 0, 1) # Clipping to valid area
    if valid:
        pad_count = tf.maximum(sampling_count - tf.shape(valid_indices)[0], 0)
        proposal = tf.pad(proposal, [[0, pad_count], [0, 0]])
    proposal = tf.stop_gradient(proposal)
    return proposal

class Classifier2Proposal(tf.keras.layers.Layer):
    def __init__(self, valid = True, batch_size = 1, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, **kwargs):
        super(Classifier2Proposal, self).__init__(**kwargs)
        self.valid = valid
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.clip_ratio = clip_ratio

    def call(self, inputs):
        cls_logits, cls_regress, proposals = inputs[:3]
        out = map_fn(classifier2proposal, cls_logits, cls_regress, proposals, dtype = proposals.dtype, batch_size = self.batch_size, valid = self.valid, mean = self.mean, std = self.std, clip_ratio = self.clip_ratio)
        return out
    
    def get_config(self):
        config = super(Classifier2Proposal, self).get_config()
        config["valid"] = self.valid
        config["batch_size"] = self.batch_size
        config["mean"] = self.mean
        config["std"] = self.std
        config["clip_ratio"] = self.clip_ratio
        return config

class FusedSemanticHead(tf.keras.layers.Layer):
    def __init__(self, n_class = 21, n_feature = 256, n_depth = 4, method = "bilinear", normalize = None, activation = tf.keras.activations.relu,  **kwargs):
        """
        Multi-level fused semantic segmentation head.(https://github.com/open-mmlab/mmdetection/blob/ff9bc39913cb3ff5dde79d3933add7dc2561bab7/mmdet/models/roi_heads/mask_heads/fused_semantic_head.py)
        """
        super(FusedSemanticHead, self).__init__(**kwargs)   
        self.n_class = n_class
        self.n_feature = n_feature
        self.n_depth = n_depth
        self.method = method
        self.normalize = normalize
        self.activation = activation

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        ch = input_shape[0][-1]
        
        self.lateral_convs = []
        for index in range(len(input_shape)):
            conv = [tf.keras.layers.Conv2D(ch, 1, use_bias = self.normalize is None, name = "lateral_conv{0}".format(index + 1))]
            if self.normalize is not None:
                conv.append(self.normalize(name = "lateral_bn{0}".format(index + 1)))
            if self.activation is not None:
                conv.append(tf.keras.layers.Activation(self.activation, name = "lateral_act{0}".format(index + 1)))
            self.lateral_convs.append(conv)
        
        self.convs = []
        for index in range(self.n_depth):
            self.convs.append(tf.keras.layers.Conv2D(self.n_feature if index != 0 else ch, 3, padding = "same", use_bias = self.normalize is None, name = "feature_conv{0}".format(index + 1)))
            if self.normalize is not None:
                self.convs.append(self.normalize(axis = -1, name = "feature_bn{0}".format(index + 1)))
            if self.activation is not None:
                self.convs.append(tf.keras.layers.Activation(self.activation, name = "feature_act{0}".format(index + 1)))
        
        self.embed = [tf.keras.layers.Conv2D(self.n_feature, 1, use_bias = self.normalize is None, name = "embed_conv")]
        if self.normalize is not None:
            self.embed.append(self.normalize(axis = -1, name = "embed_bn"))
        if self.activation is not None:
            self.embed.append(tf.keras.layers.Activation(self.activation, name = "embed_act"))
            
        self.logits = tf.keras.layers.Conv2D(self.n_class, 1, use_bias = True, name = "logits")
        
    def call(self, inputs, level = 1, feature = False):
        if not isinstance(inputs, list):
            inputs = [inputs]
        out = inputs[level]
        for layer in self.lateral_convs[level]:
            out = layer(out)
        target_size = tf.shape(out)[-3:-1]
        for i, x in enumerate(inputs):
            if i != level:
                x = tf.image.resize(x, target_size, method = self.method, name = "resample{0}".format(i + 1))
                for layer in self.lateral_convs[i]:
                    x = layer(x)
                out = out + x
        
        for layer in self.convs:
            out = layer(out)
            
        logits = self.logits(out)
        if feature:
            for layer in self.embed:
                out = layer(out)
            return logits, out
        else:
            return logits
    
    def get_config(self):
        config = super(FusedSemanticHead, self).get_config()
        config["n_class"] = self.n_class
        config["n_feature"] = self.n_feature
        config["n_depth"] = self.n_depth
        config["method"] = self.method
        config["normalize"] = self.normalize
        config["activation"] = self.activation
        return config

def roi2level(bbox, n_level, input_shape = (224, 224)):
    if 1 < tf.reduce_max(bbox):
        bbox = tf.divide(bbox, tf.cast(tf.tile(input_shape[::-1], [2]), bbox.dtype))
    x1, y1, x2, y2 = tf.split(bbox, 4, axis = -1)
    h = y2 - y1
    w = x2 - x1

    bbox_area = h * w
    image_area = tf.cast(input_shape[0] * input_shape[1], bbox.dtype)

    roi_level = tf.cast(tf.floor(tf.math.log((tf.sqrt(bbox_area)) / ((56. / tf.sqrt(image_area)) + 1e-6)) / tf.math.log(2.)), tf.int32)
    roi_level = tf.clip_by_value(roi_level, 0, n_level - 1)
    roi_level = tf.squeeze(roi_level, axis = -1)
    return roi_level