import tensorflow as tf
import numpy as np

from tfdet.core.anchor import generate_anchors
from tfdet.core.bbox import delta2bbox
from tfdet.core.ops import RoiAlign
from tfdet.core.util import map_fn
from tfdet.model.postprocess.anchor import FilterDetection

def rpn_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

def roi_conv(filters, kernel_size, strides = 1, padding = "same", use_bias = True, kernel_initializer = "he_normal", **kwargs):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, **kwargs)

class RegionProposalNetwork(tf.keras.layers.Layer):
    def __init__(self, n_anchor, n_feature = 512, use_bias = None, feature_share = True, concat = False,
                 convolution = rpn_conv, normalize = None, activation = tf.keras.activations.relu, **kwargs):
        super(RegionProposalNetwork, self).__init__(**kwargs)
        self.n_anchor = n_anchor
        self.n_feature = n_feature
        self.use_bias = (normalize is None) if use_bias is None else use_bias
        self.feature_share = feature_share
        self.concat = concat
        self.convolution = convolution
        self.normalize = normalize
        self.activation = activation

        self.score_reshape = tf.keras.layers.Reshape((-1, 1), name = "score")
        self.delta = tf.keras.layers.Reshape((-1, 4), name = "delta")
        self.score_act = tf.keras.layers.Activation(tf.keras.activations.sigmoid, dtype = tf.float32, name = "score_act")
        self.delta_act = tf.keras.layers.Activation(tf.keras.activations.linear, dtype = tf.float32, name = "delta_act")

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        if self.feature_share:
            self.feature = [self.convolution(self.n_feature, 3, padding = "same", use_bias = self.use_bias, name = "shared_feature_conv")] * len(input_shape)
            if self.activation is not None:
                self.feature_act = [tf.keras.layers.Activation(self.activation, name = "shared_feature_conv_act")] * len(input_shape)
            self.score = [self.convolution(self.n_anchor, 1, use_bias = True, name = "shared_score_conv")] * len(input_shape)
            self.regress = [self.convolution(self.n_anchor * 4, 1, use_bias = True, name = "shared_regress_conv")] * len(input_shape)
        else:
            self.feature = [self.convolution(self.n_feature, 3, padding = "same", use_bias = self.use_bias, activation = self.activation, name = "feature_conv{0}".format(index + 1)) for index in range(len(input_shape))]
            if self.activation is not None:
                self.feature_act = [tf.keras.layers.Activation(self.activation, name = "feature_conv_act{0}".format(index + 1)) for index in range(len(input_shape))]
            self.score = [self.convolution(self.n_anchor, 1, use_bias = True, name = "score_conv{0}".format(index + 1)) for index in range(len(input_shape))]
            self.regress = [self.convolution(self.n_anchor * 4, 1, use_bias = True, name = "regress_conv{0}".format(index + 1)) for index in range(len(input_shape))]
        if self.normalize is not None:
            self.norm = [self.normalize(name = "feature_norm{0}".format(index + 1)) for index in range(len(input_shape))]
        
        if self.concat and 1 < len(input_shape):
            self.score_concat = tf.keras.layers.Concatenate(axis = -2, name = "score_concat")
            self.delta_concat = tf.keras.layers.Concatenate(axis = -2, name = "delta_concat")

    def call(self, inputs):
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        
        out = []
        for i, x in enumerate(inputs):
            feature = self.feature[i](x)
            if self.activation is not None:
                feature = self.feature_act[i](feature)
            if self.normalize is not None:
                feature = self.norm[i](feature)
            score = self.score_reshape(self.score[i](feature))
            delta = self.delta(self.regress[i](feature))
            out.append([score, delta])
        out = list(zip(*out))
        if len(out[0]) == 1:
            out = [o[0] for o in out]
        elif self.concat:
            out[0] = self.score_concat(out[0])
            out[1] = self.delta_concat(out[1])
        if isinstance(out[0], (tuple, list)):
            out[0] = [self.score_act(o) for o in out[0]]
            out[1] = [self.delta_act(o) for o in out[1]]
        else:
            out[0] = self.score_act(out[0])
            out[1] = self.delta_act(out[1])
        return out
    
    def get_config(self):
        config = super(RegionProposalNetwork, self).get_config()
        config["n_anchor"] = self.n_anchor
        config["n_feature"] = self.n_feature
        config["use_bias"] = self.use_bias
        config["feature_share"] = self.feature_share
        config["concat"] = self.concat
        return config

class RoiClassifier(tf.keras.layers.Layer):
    def __init__(self, n_class = 21, n_feature = 1024, use_bias = None, convolution = roi_conv, normalize = None, activation = tf.keras.activations.relu, **kwargs):
        super(RoiClassifier, self).__init__(**kwargs)   
        self.n_class = n_class
        self.n_feature = n_feature
        self.use_bias = (normalize is None) if use_bias is None else use_bias
        self.convolution = convolution
        self.normalize = normalize
        self.activation = activation
        
        self.norm1 = self.norm2 = None
        if self.normalize is not None:
            self.norm1 = self.normalize(name = "pooling_norm")
        self.act1 = tf.keras.layers.Activation(activation, name = "pooling_act")
        self.conv2 = self.convolution(n_feature, 1, use_bias = self.use_bias, name = "feature_conv")
        if self.normalize is not None:
            self.norm2 = self.normalize(name = "feature_norm")
        self.act2 = tf.keras.layers.Activation(activation, name = "feature_act")
        self.feature = tf.keras.layers.Reshape([-1, n_feature], name = "shared_feature")
        self.logits = tf.keras.layers.Dense(n_class, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01), name = "logits")
        self.regress = tf.keras.layers.Dense(n_class * 4, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.001), name = "regress")
        self.delta = tf.keras.layers.Reshape([-1, n_class, 4], name = "delta")
        self.logits_act = tf.keras.layers.Activation(tf.keras.activations.softmax, dtype = tf.float32, name = "logits_act")
        self.delta_act = tf.keras.layers.Activation(tf.keras.activations.linear, dtype = tf.float32, name = "delta_act")

    def build(self, input_shape):
        self.conv1 = self.convolution(self.n_feature, input_shape[-3:-1], padding = "valid", use_bias = self.use_bias, name = "pooling_conv")

    def call(self, inputs):
        out = inputs
        roi_shape = tf.shape(out)
        batch_size = roi_shape[0]
        num_proposals = roi_shape[1]
        pool_size, roi_feature = tf.keras.backend.int_shape(out)[-2:]
        out = tf.reshape(out, [-1, pool_size, pool_size, roi_feature])
        
        for layer in [self.conv1, self.norm1, self.act1, 
                      self.conv2, self.norm2, self.act2]:
            if layer is not None:
                out = layer(out)
        out = self.feature(out)
        logits = self.logits(out)
        delta = self.delta(self.regress(out))
        logits = self.logits_act(logits)
        delta = self.delta_act(delta)
        y_pred = tf.reshape(logits, [batch_size, num_proposals, self.n_class])
        bbox_pred = tf.reshape(delta, [batch_size, num_proposals, self.n_class, 4])
        return y_pred, bbox_pred

    def get_config(self):
        config = super(RoiClassifier, self).get_config()
        config["n_class"] = self.n_class
        config["n_feature"] = self.n_feature
        config["use_bias"] = self.use_bias
        return config

class RoiMask(tf.keras.layers.Layer):
    def __init__(self, n_class = 21, n_feature = 256, n_depth = 4, scale = 2, use_bias = None, convolution = roi_conv, normalize = None, activation = tf.keras.activations.relu, **kwargs):
        super(RoiMask, self).__init__(**kwargs)   
        self.n_class = n_class
        self.n_feature = n_feature
        self.n_depth = n_depth
        self.scale = scale
        self.use_bias = (normalize is None) if use_bias is None else use_bias
        self.convolution = convolution
        self.normalize = normalize
        self.activation = activation
        
        self.layers = []        
        for index in range(n_depth):
            self.layers.append(self.convolution(n_feature, 3, padding = "same", use_bias = self.use_bias, name = "feature_conv{0}".format(index + 1)))
            if self.normalize is not None:
                self.layers.append(self.normalize(name = "feature_norm{0}".format(index + 1)))
            self.layers.append(tf.keras.layers.Activation(activation, name = "feature_act{0}".format(index + 1)))
        self.deconv = tf.keras.layers.Conv2DTranspose(n_feature, (scale, scale), strides = scale, activation = activation, kernel_initializer = "he_normal", name = "deconv")
        self.mask = self.convolution(n_class, 1, name = "mask")
        self.mask_act = tf.keras.layers.Activation(tf.keras.activations.sigmoid, dtype = tf.float32, name = "mask_act")
        
    def build(self, input_shape):
        input_shape = [input_shape] if not isinstance(input_shape, (tuple, list)) else input_shape
        if 1 < len(input_shape):
            self.resample = self.convolution(input_shape[0][-1], 1, use_bias = True, name = "resample_conv")
        
    def call(self, inputs, feature = False):
        inputs = [inputs] if not isinstance(inputs, (tuple, list)) else inputs
        out = inputs[0]
        if 1 < len(inputs) and inputs[1] is not None:
            out = out + self.resample(inputs[1])
        
        roi_shape = tf.shape(out)
        batch_size = roi_shape[0]
        num_proposals = roi_shape[1]
        pool_size, roi_feature = tf.keras.backend.int_shape(out)[-2:]
        mask_size = int(pool_size * self.scale)
        out = tf.reshape(out, [-1, pool_size, pool_size, roi_feature])
        
        for layer in self.layers:
            out = layer(out)
        deconv = self.deconv(out)
        mask_pred = self.mask_act(self.mask(deconv))
        
        mask_pred = tf.reshape(mask_pred, [batch_size, num_proposals, mask_size, mask_size, self.n_class])
        if feature:
            out = tf.reshape(out, [batch_size, num_proposals, pool_size, pool_size, roi_feature])
            return mask_pred, out  
        else:
            return mask_pred    
            
    def get_config(self):
        config = super(RoiMask, self).get_config()
        config["n_class"] = self.n_class
        config["n_feature"] = self.n_feature
        config["n_depth"] = self.n_depth
        config["scale"] = self.scale
        config["use_bias"] = self.use_bias
        return config
    
class Rpn2Proposal(FilterDetection):
    def __init__(self, proposal_count = 1000, iou_threshold = 0.7, score_threshold = float('-inf'), soft_nms = False, valid_inside_anchor = False, performance_count = 5000,
                 mean = [0., 0., 0., 0.], std = [1., 1., 1., 1.], clip_ratio = 16 / 1000,
                 batch_size = 1, dtype = tf.float32, **kwargs):
        super(Rpn2Proposal, self).__init__(proposal_count = proposal_count, iou_threshold = iou_threshold, score_threshold = score_threshold, soft_nms = soft_nms, valid_inside_anchor = valid_inside_anchor, ignore_label = None, performance_count = performance_count,
                                           mean = mean, std = std, clip_ratio = clip_ratio,
                                           batch_size = batch_size, dtype = dtype, **kwargs)   

    def call(self, inputs):
        y_pred, bbox_pred = super(Rpn2Proposal, self).call(inputs)
        y_pred = tf.stop_gradient(y_pred)
        bbox_pred = tf.stop_gradient(bbox_pred)
        return bbox_pred

class Classifier2Proposal(tf.keras.layers.Layer):
    def __init__(self, mean = [0., 0., 0., 0.], std = [0.1, 0.1, 0.2, 0.2], clip_ratio = 16 / 1000, dtype = tf.float32, **kwargs):
        kwargs["dtype"] = dtype
        super(Classifier2Proposal, self).__init__(**kwargs)
        self.mean = mean
        self.std = std
        self.clip_ratio = clip_ratio

    def call(self, inputs):
        y_pred, bbox_pred, proposals = inputs[:3]
        
        label_pred = tf.expand_dims(tf.argmax(y_pred, axis = -1, output_type = tf.int32), axis = -1)
        delta_pred = tf.gather_nd(bbox_pred, label_pred, batch_dims = 2)
        
        proposals = delta2bbox(proposals, delta_pred, mean = self.mean, std = self.std, clip_ratio = self.clip_ratio)
        proposals = tf.clip_by_value(proposals, 0, 1)
        return tf.stop_gradient(proposals)
    
    def get_config(self):
        config = super(Classifier2Proposal, self).get_config()
        config["mean"] = self.mean
        config["std"] = self.std
        config["clip_ratio"] = self.clip_ratio
        return config

class FusedSemanticHead(tf.keras.layers.Layer):
    def __init__(self, n_class = 21, n_feature = 256, n_depth = 4, use_bias = None, method = "bilinear", logits_activation = tf.keras.activations.softmax, convolution = roi_conv, normalize = None, activation = tf.keras.activations.relu,  **kwargs):
        """
        Multi-level fused semantic segmentation head.(https://github.com/open-mmlab/mmdetection/blob/ff9bc39913cb3ff5dde79d3933add7dc2561bab7/mmdet/models/roi_heads/mask_heads/fused_semantic_head.py)
        """
        super(FusedSemanticHead, self).__init__(**kwargs)   
        self.n_class = n_class
        self.n_feature = n_feature
        self.n_depth = n_depth
        self.use_bias = (normalize is None) if use_bias is None else use_bias
        self.method = method
        self.logits_activation = logits_activation
        self.convolution = convolution
        self.normalize = normalize
        self.activation = activation

    def build(self, input_shape):
        if not isinstance(input_shape, (tuple, list)):
            input_shape = [input_shape]
        ch = input_shape[0][-1]
        
        self.lateral_convs = []
        for index in range(len(input_shape)):
            conv = [self.convolution(ch, 1, use_bias = self.use_bias, name = "lateral_conv{0}".format(index + 1))]
            if self.normalize is not None:
                conv.append(self.normalize(name = "lateral_norm{0}".format(index + 1)))
            conv.append(tf.keras.layers.Activation(self.activation, name = "lateral_act{0}".format(index + 1)))
            self.lateral_convs.append(conv)
        
        self.convs = []
        for index in range(self.n_depth):
            self.convs.append(self.convolution(self.n_feature if index != 0 else ch, 3, padding = "same", use_bias = self.use_bias, name = "feature_conv{0}".format(index + 1)))
            if self.normalize is not None:
                self.convs.append(self.normalize(axis = -1, name = "feature_norm{0}".format(index + 1)))
            self.convs.append(tf.keras.layers.Activation(self.activation, name = "feature_act{0}".format(index + 1)))
        
        self.embed = [self.convolution(self.n_feature, 1, use_bias = self.use_bias, name = "embed_conv")]
        if self.normalize is not None:
            self.embed.append(self.normalize(axis = -1, name = "embed_norm"))
        self.embed.append(tf.keras.layers.Activation(self.activation, name = "embed_act"))
            
        self.logits = tf.keras.layers.Conv2D(self.n_class, 1, use_bias = True, kernel_initializer = "he_normal", name = "logits")
        self.logits_act = tf.keras.layers.Activation(self.logits_activation if self.logits_activation is not None else tf.keras.activations.linear, dtype = tf.float32, name = "logits_act")
        
    def call(self, inputs, level = 1, feature = False):
        if not isinstance(inputs, (tuple, list)):
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
            
        logits = self.logits_act(self.logits(out))
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
        config["use_bias"] = self.use_bias
        config["method"] = self.method
        return config

def rpn_head(feature, image_shape = [1024, 1024],
             scale = [32, 64, 128, 256, 512], ratio = [0.5, 1, 2], octave = 1,
             n_feature = 256, use_bias = None, feature_share = True,
             convolution = rpn_conv, normalize = None, activation = tf.keras.activations.relu):
    if tf.is_tensor(image_shape) and 2 < tf.keras.backend.ndim(image_shape) or (not tf.is_tensor(image_shape) and 2 < np.ndim(image_shape)):
        image_shape = tf.shape(image_shape) if tf.keras.backend.int_shape(image_shape)[-3] is None else tf.keras.backend.int_shape(image_shape)
    if 2 < np.shape(image_shape)[0]:
        image_shape = image_shape[-3:-1]
    if not isinstance(feature, (tuple, list)):
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
    y_pred, bbox_pred = RegionProposalNetwork(n_anchor, n_feature = n_feature, use_bias = use_bias, feature_share = feature_share, concat = False, convolution = convolution, normalize = normalize, activation = activation, name = "region_proposal_network")(feature)
    anchors = generate_anchors(feature, image_shape, scale, ratio, normalize = True, auto_scale = True, concat = False, dtype = tf.float32)
    return y_pred, bbox_pred, anchors

def rcnn_head(feature, proposals, mask_feature = None, semantic_feature = None,
              n_class = 21, image_shape = [1024, 1024],
              classifier = True, mask = False,
              cls_n_feature = 1024, mask_n_feature = 256, mask_n_depth = 4, mask_scale = 2, use_bias = None,
              pool_size = 7, method = "bilinear",
              cls_convolution = roi_conv, cls_normalize = None, cls_activation = tf.keras.activations.relu,
              mask_convolution = roi_conv, mask_normalize = None, mask_activation = tf.keras.activations.relu):
    if tf.is_tensor(image_shape) and 2 < tf.keras.backend.ndim(image_shape) or (not tf.is_tensor(image_shape) and 2 < np.ndim(image_shape)):
        image_shape = tf.shape(image_shape) if tf.keras.backend.int_shape(image_shape)[-3] is None else tf.keras.backend.int_shape(image_shape)
    if 2 < np.shape(image_shape)[0]:
        image_shape = image_shape[-3:-1]
    if not isinstance(feature, (tuple, list)):
        feature = [feature]
    feature = list(feature)
    
    roi_extractor = RoiAlign(pool_size, method, dtype = tf.float32)
    roi = roi_extractor([feature, proposals], image_shape)
    if semantic_feature is not None:
        semantic_roi_extractor = RoiAlign(pool_size, method, dtype = tf.float32)
        semantic_roi = semantic_roi_extractor([semantic_feature, proposals], image_shape)
        roi = tf.keras.layers.Add()([roi, semantic_roi])
    
    y_pred = bbox_pred = mask_pred = mask_feature = None
    if classifier:
        y_pred, bbox_pred = RoiClassifier(n_class, cls_n_feature, use_bias, convolution = cls_convolution, normalize = cls_normalize, activation = cls_activation)(roi)
    if mask:
        mask_pred, mask_feature = RoiMask(n_class, mask_n_feature, mask_n_depth, mask_scale, use_bias, convolution = mask_convolution, normalize = mask_normalize, activation = mask_activation)([roi, mask_feature], feature = True)
    result = [r for r in [y_pred, bbox_pred, mask_pred, mask_feature] if r is not None]
    if len(result) == 1:
        result = result[0]
    elif len(result) == 0:
        result = None
    return result

def bbox_head(feature, proposals, semantic_feature = None,
              n_class = 21, image_shape = [1024, 1024], n_feature = 1024, use_bias = None,
              pool_size = 7, method = "bilinear",
              convolution = roi_conv, normalize = None, activation = tf.keras.activations.relu):
    return rcnn_head(feature, proposals, None, semantic_feature,
                     n_class = n_class, image_shape = image_shape, cls_n_feature = n_feature, use_bias = use_bias,
                     classifier = True, mask = False,
                     pool_size = pool_size, method = method,
                     cls_convolution = convolution, cls_normalize = normalize, cls_activation = activation)

def mask_head(feature, proposals, mask_feature = None, semantic_feature = None,
              n_class = 21, image_shape = [1024, 1024], n_feature = 256, n_depth = 4, scale = 2, use_bias = None,
              pool_size = 14, method = "bilinear",
              convolution = roi_conv, normalize = None, activation = tf.keras.activations.relu):
    return rcnn_head(feature, proposals, mask_feature, semantic_feature,
                     n_class = n_class, image_shape = image_shape, mask_n_feature = n_feature, mask_n_depth = n_depth, mask_scale = scale, use_bias = use_bias,
                     classifier = False, mask = True,
                     pool_size = pool_size, method = method,
                     mask_convolution = convolution, mask_normalize = normalize, mask_activation = activation)

def semantic_head(feature, n_class = 21, level = 1, n_feature = 256, n_depth = 4, use_bias = None, method = "bilinear",
                  logits_activation = tf.keras.activations.softmax, convolution = roi_conv, normalize = None, activation = tf.keras.activations.relu):
    if not isinstance(feature, (tuple, list)):
        feature = [feature]
    feature = list(feature)
    return FusedSemanticHead(n_class, n_feature, n_depth, use_bias = use_bias, method = method, logits_activation = logits_activation, convolution = convolution, normalize = normalize, activation = activation)(feature, min(level, len(feature) - 1), feature = True)