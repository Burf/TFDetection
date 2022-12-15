import tensorflow as tf
import numpy as np

from tfdet.core.anchor import generate_anchors
from tfdet.core.bbox import delta2bbox
from tfdet.core.roi import roi_align
from tfdet.core.util import pad_nms, map_fn

def conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

class RegionProposalNetwork(tf.keras.layers.Layer):
    def __init__(self, n_anchor, n_feature = 512, use_bias = True, feature_share = True, convolution = conv, normalize = None, activation = tf.keras.activations.relu, **kwargs):
        super(RegionProposalNetwork, self).__init__(**kwargs)
        self.n_anchor = n_anchor
        self.n_feature = n_feature
        self.use_bias = use_bias
        self.feature_share = feature_share
        self.convolution = convolution
        self.normalize = normalize
        self.activation = activation

        self.score_reshape = tf.keras.layers.Reshape((-1, 1), name = "score")
        self.delta = tf.keras.layers.Reshape((-1, 4), name = "delta")

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        if self.feature_share:
            self.feature = [self.convolution(self.n_feature, 3, padding = "same", use_bias = False, activation = self.activation, name = "shared_feature_conv")] * len(input_shape)
            self.score = [self.convolution(self.n_anchor, 1, use_bias = self.use_bias, activation = tf.keras.activations.sigmoid, name = "shared_score_conv")] * len(input_shape)
            self.regress = [self.convolution(self.n_anchor * 4, 1, use_bias = self.use_bias, name = "shared_regress_conv")] * len(input_shape)
        else:
            self.feature = [self.convolution(self.n_feature, 3, padding = "same", use_bias = False, activation = self.activation, name = "feature_conv{0}".format(index + 1)) for index in range(len(input_shape))]
            self.score = [self.convolution(self.n_anchor, 1, use_bias = self.use_bias, activation = tf.keras.activations.sigmoid, name = "score_conv{0}".format(index + 1)) for index in range(len(input_shape))]
            self.regress = [self.convolution(self.n_anchor * 4, 1, use_bias = self.use_bias, name = "regress_conv{0}".format(index + 1)) for index in range(len(input_shape))]
        if self.normalize is not None:
            self.norm = [self.normalize(name = "feature_norm{0}".format(index + 1)) for index in range(len(input_shape))]
        
        if 1 < len(input_shape):
            self.score_concat = tf.keras.layers.Concatenate(axis = -2, name = "score_concat")
            self.delta_concat = tf.keras.layers.Concatenate(axis = -2, name = "delta_concat")

    def call(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        out = []
        for i, x in enumerate(inputs):
            feature = self.feature[i](x)
            if self.normalize is not None:
                feature = self.norm[i](feature)
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
        config["n_feature"] = self.n_feature
        config["use_bias"] = self.use_bias
        config["feature_share"] = self.feature_share
        return config
    
class Rpn2Proposal(tf.keras.layers.Layer):
    def __init__(self, proposal_count = 1000, iou_threshold = 0.7, soft_nms = False, valid = False, performance_count = 5000, batch_size = 1, mean = [0., 0., 0., 0.], std = [1., 1., 1., 1.], clip_ratio = 16 / 1000, **kwargs):
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
        proposals = map_fn(pad_nms, proposals, rpn_score, dtype = rpn_score.dtype, batch_size = self.batch_size, 
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
    def __init__(self, pool_size = 7, method = "bilinear", batch_size = 1, **kwargs):
        super(RoiAlign, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.method = method
        self.batch_size = batch_size

    def call(self, inputs, image_shape = [1024, 1024]):
        proposals, feature = inputs
        if not isinstance(feature, (tuple, list)):
            feature = [feature]
        out = map_fn(roi_align, proposals, *feature, dtype = proposals.dtype, batch_size = self.batch_size, image_shape = image_shape, pool_size = self.pool_size, method = self.method)
        return out
    
    def get_config(self):
        config = super(RoiAlign, self).get_config()
        config["pool_size"] = self.pool_size
        config["method"] = self.method
        config["batch_size"] = self.batch_size
        return config

class RoiClassifier(tf.keras.layers.Layer):
    def __init__(self, n_class = 21, n_feature = 1024, convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu, **kwargs):
        super(RoiClassifier, self).__init__(**kwargs)   
        self.n_class = n_class
        self.n_feature = n_feature
        self.convolution = convolution
        self.normalize = normalize
        self.activation = activation

        self.norm1 = self.norm2 = None
        if self.normalize is not None:
            self.norm1 = tf.keras.layers.TimeDistributed(self.normalize(), name = "pooling_norm")
        self.act1 = tf.keras.layers.Activation(activation, name = "pooling_act")
        self.conv2 = tf.keras.layers.TimeDistributed(self.convolution(n_feature, 1, use_bias = True), name = "feature_conv")
        if self.normalize is not None:
            self.norm2 = tf.keras.layers.TimeDistributed(self.normalize(), name = "feature_norm")
        self.act2 = tf.keras.layers.Activation(activation, name = "feature_act")
        self.feature = tf.keras.layers.Reshape([-1, n_feature], name = "shared_feature")
        self.logits = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_class, activation = tf.keras.activations.softmax), name = "logits")
        self.regress = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_class * 4), name = "regress")
        self.delta = tf.keras.layers.Reshape([-1, n_class, 4], name = "delta")

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.TimeDistributed(self.convolution(self.n_feature, input_shape[-3:-1], padding = "valid", use_bias = True), name = "pooling_conv")

    def call(self, inputs):
        out = inputs
        for layer in [self.conv1, self.norm1, self.act1, 
                      self.conv2, self.norm2, self.act2]:
            if layer is not None:
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
        return config

class RoiMask(tf.keras.layers.Layer):
    def __init__(self, n_class = 21, n_feature = 256, n_depth = 4, convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu, **kwargs):
        super(RoiMask, self).__init__(**kwargs)   
        self.n_class = n_class
        self.n_feature = n_feature
        self.n_depth = n_depth
        self.convolution = convolution
        self.normalize = normalize
        self.activation = activation
        
        self.layers = []        
        for index in range(n_depth):
            self.layers.append(tf.keras.layers.TimeDistributed(self.convolution(n_feature, 3, padding = "same", use_bias = True), name = "feature_conv{0}".format(index + 1)))
            if self.normalize is not None:
                self.layers.append(tf.keras.layers.TimeDistributed(self.normalize(), name = "feature_norm{0}".format(index + 1)))
            self.layers.append(tf.keras.layers.Activation(activation, name = "feature_act{0}".format(index + 1)))
        self.deconv = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(256, (2, 2), strides = 2, activation = activation, kernel_initializer = "he_normal"), name = "deconv")
        self.mask = tf.keras.layers.TimeDistributed(self.convolution(n_class, 1, activation = tf.keras.activations.sigmoid), name = "mask")
        
    def build(self, input_shape):
        if isinstance(input_shape, list):
            self.resample = tf.keras.layers.TimeDistributed(self.convolution(input_shape[0][-1], 1, use_bias = True), name = "resample_conv")

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
        return config

def classifier2proposal(cls_logit, cls_regress, proposal, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000):
    sampling_count = tf.shape(proposal)[0]
    valid_indices = tf.where(0 < tf.reduce_max(proposal, axis = -1))
    cls_logit = tf.gather_nd(cls_logit, valid_indices)
    cls_regress = tf.gather_nd(cls_regress, valid_indices)
    proposal = tf.gather_nd(proposal, valid_indices)

    logit_indices = tf.argmax(cls_logit, axis = -1, output_type = tf.int32)
    indices = tf.stack([tf.range(tf.shape(cls_logit)[0]), logit_indices], axis = -1)
    delta = tf.gather_nd(cls_regress, indices)
    proposal = tf.stop_gradient(proposal)
    delta = tf.stop_gradient(delta)

    proposal = delta2bbox(proposal, delta, mean, std, clip_ratio) # Transform delta to bbox
    proposal = tf.clip_by_value(proposal, 0, 1) # Clipping to valid area
    pad_count = tf.maximum(sampling_count - tf.shape(valid_indices)[0], 0)
    proposal = tf.pad(proposal, [[0, pad_count], [0, 0]])
    return proposal

class Classifier2Proposal(tf.keras.layers.Layer):
    def __init__(self, batch_size = 1, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, **kwargs):
        super(Classifier2Proposal, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.clip_ratio = clip_ratio

    def call(self, inputs):
        cls_logits, cls_regress, proposals = inputs[:3]
        out = map_fn(classifier2proposal, cls_logits, cls_regress, proposals, dtype = proposals.dtype, batch_size = self.batch_size, mean = self.mean, std = self.std, clip_ratio = self.clip_ratio)
        out = tf.stop_gradient(out)
        return out
    
    def get_config(self):
        config = super(Classifier2Proposal, self).get_config()
        config["batch_size"] = self.batch_size
        config["mean"] = self.mean
        config["std"] = self.std
        config["clip_ratio"] = self.clip_ratio
        return config

class FusedSemanticHead(tf.keras.layers.Layer):
    def __init__(self, n_class = 21, n_feature = 256, n_depth = 4, method = "bilinear", logits_activation = None, convolution = conv, normalize = None, activation = tf.keras.activations.relu,  **kwargs):
        """
        Multi-level fused semantic segmentation head.(https://github.com/open-mmlab/mmdetection/blob/ff9bc39913cb3ff5dde79d3933add7dc2561bab7/mmdet/models/roi_heads/mask_heads/fused_semantic_head.py)
        """
        super(FusedSemanticHead, self).__init__(**kwargs)   
        self.n_class = n_class
        self.n_feature = n_feature
        self.n_depth = n_depth
        self.method = method
        self.logits_activation = logits_activation
        self.convolution = convolution
        self.normalize = normalize
        self.activation = activation

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        ch = input_shape[0][-1]
        
        self.lateral_convs = []
        for index in range(len(input_shape)):
            conv = [self.convolution(ch, 1, use_bias = self.normalize is None, name = "lateral_conv{0}".format(index + 1))]
            if self.normalize is not None:
                conv.append(self.normalize(name = "lateral_norm{0}".format(index + 1)))
            conv.append(tf.keras.layers.Activation(self.activation, name = "lateral_act{0}".format(index + 1)))
            self.lateral_convs.append(conv)
        
        self.convs = []
        for index in range(self.n_depth):
            self.convs.append(self.convolution(self.n_feature if index != 0 else ch, 3, padding = "same", use_bias = self.normalize is None, name = "feature_conv{0}".format(index + 1)))
            if self.normalize is not None:
                self.convs.append(self.normalize(axis = -1, name = "feature_norm{0}".format(index + 1)))
            self.convs.append(tf.keras.layers.Activation(self.activation, name = "feature_act{0}".format(index + 1)))
        
        self.embed = [self.convolution(self.n_feature, 1, use_bias = self.normalize is None, name = "embed_conv")]
        if self.normalize is not None:
            self.embed.append(self.normalize(axis = -1, name = "embed_norm"))
        self.embed.append(tf.keras.layers.Activation(self.activation, name = "embed_act"))
            
        self.logits = tf.keras.layers.Conv2D(self.n_class, 1, use_bias = True, activation = self.logits_activation, kernel_initializer = "he_normal", name = "logits")
        
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
        return config

def rpn_head(feature, image_shape = [1024, 1024],
             scale = [32, 64, 128, 256, 512], ratio = [0.5, 1, 2], octave = 1,
             n_feature = 256, use_bias = True, feature_share = True,
             convolution = conv, normalize = None, activation = tf.keras.activations.relu):
    if tf.is_tensor(image_shape) and 2 < tf.keras.backend.ndim(image_shape) or (not tf.is_tensor(image_shape) and 2 < np.ndim(image_shape)):
        image_shape = tf.shape(image_shape) if tf.keras.backend.int_shape(image_shape)[-3] is None else tf.keras.backend.int_shape(image_shape)
    if 2 < np.shape(image_shape)[0]:
        image_shape = image_shape[-3:-1]
    if not isinstance(feature, list):
        feature = [feature]
    if np.ndim(scale) == 0:
        scale = [[scale]]
    elif np.ndim(scale) == 1:
        scale = np.expand_dims(scale, axis = -1)
    if np.ndim(ratio) == 0:
        ratio = [ratio]
    feature = list(feature)
    
    if np.ndim(scale) == 2 and np.shape(scale)[-1] == 1:
        scale = np.multiply(scale, [[2 ** (o / octave) for o in range(octave)]])
    n_anchor = len(scale) * len(ratio)
    if (len(feature) % len(scale)) == 0:
        n_anchor = len(scale[0]) * len(ratio)
    score, regress = RegionProposalNetwork(n_anchor, n_feature = n_feature, use_bias = use_bias, feature_share = feature_share, convolution = convolution, normalize = normalize, activation = activation, name = "region_proposal_network")(feature)
    anchors = generate_anchors(feature, image_shape, scale, ratio, normalize = True, auto_scale = True, dtype = score.dtype)
    return score, regress, anchors

def rcnn_head(feature, proposals, mask_feature = None, semantic_feature = None,
              n_class = 21, image_shape = [1024, 1024],
              classifier = True, mask = False,
              cls_n_feature = 1024, mask_n_feature = 256, mask_n_depth = 4,
              pool_size = 7, semantic_pool_size = 14, method = "bilinear",
              cls_convolution = conv, cls_normalize = tf.keras.layers.BatchNormalization, cls_activation = tf.keras.activations.relu,
              mask_convolution = conv, mask_normalize = tf.keras.layers.BatchNormalization, mask_activation = tf.keras.activations.relu):
    if tf.is_tensor(image_shape) and 2 < tf.keras.backend.ndim(image_shape) or (not tf.is_tensor(image_shape) and 2 < np.ndim(image_shape)):
        image_shape = tf.shape(image_shape) if tf.keras.backend.int_shape(image_shape)[-3] is None else tf.keras.backend.int_shape(image_shape)
    if 2 < np.shape(image_shape)[0]:
        image_shape = image_shape[-3:-1]
    if not isinstance(feature, list):
        feature = [feature]
    feature = list(feature)
    
    roi_extractor = RoiAlign(pool_size, method)
    roi = roi_extractor([proposals, feature], image_shape)
    if semantic_feature is not None:
        semantic_roi_extractor = RoiAlign(semantic_pool_size, method)
        semantic_roi = semantic_roi_extractor([proposals, semantic_feature], image_shape)
        if pool_size != semantic_pool_size:
            semantic_roi = tf.keras.layers.TimeDistributed(tf.keras.layers.Lambda(lambda args: tf.image.resize(args, [pool_size, pool_size], method = method)))(semantic_roi)
        roi = tf.keras.layers.Add()([roi, semantic_roi])
    
    cls_logits = cls_regress = mask_regress = mask_feature = None
    if classifier:
        cls_logits, cls_regress = RoiClassifier(n_class, cls_n_feature, convolution = cls_convolution, normalize = cls_normalize, activation = cls_activation)(roi)
    if mask:
        mask_regress, mask_feature = RoiMask(n_class, mask_n_feature, mask_n_depth, convolution = mask_convolution, normalize = mask_normalize, activation = mask_activation)([roi, mask_feature], feature = True)
    result = [r for r in [cls_logits, cls_regress, mask_regress, mask_feature] if r is not None]
    if len(result) == 1:
        result = result[0]
    elif len(result) == 0:
        result = None
    return result

def bbox_head(feature, proposals, semantic_feature = None,
              n_class = 21, image_shape = [1024, 1024], n_feature = 1024,
              pool_size = 7, semantic_pool_size = 14, method = "bilinear",
              convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    return rcnn_head(feature, proposals, None, semantic_feature,
                     n_class = n_class, image_shape = image_shape, cls_n_feature = n_feature,
                     classifier = True, mask = False,
                     pool_size = pool_size, semantic_pool_size = semantic_pool_size, method = method,
                     cls_convolution = convolution, cls_normalize = normalize, cls_activation = activation)

def mask_head(feature, proposals, mask_feature = None, semantic_feature = None,
              n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4,
              pool_size = 7, semantic_pool_size = 14, method = "bilinear",
              convolution = conv, normalize = tf.keras.layers.BatchNormalization, activation = tf.keras.activations.relu):
    return rcnn_head(feature, proposals, mask_feature, semantic_feature,
                     n_class = n_class, image_shape = image_shape, cls_n_feature = n_feature,
                     classifier = False, mask = True,
                     pool_size = pool_size, semantic_pool_size = semantic_pool_size, method = method,
                     mask_convolution = convolution, mask_normalize = normalize, mask_activation = activation)

def semantic_head(feature, n_class = 21, level = 1, n_feature = 256, n_depth = 4, method = "bilinear",
                  logits_activation = None, convolution = conv, normalize = None, activation = tf.keras.activations.relu):
    if not isinstance(feature, list):
        feature = [feature]
    feature = list(feature)
    return FusedSemanticHead(n_class, n_feature, n_depth, method = method, logits_activation = logits_activation, convolution = convolution, normalize = normalize, activation = activation)(feature, min(level, len(feature) - 1), feature = True)