import tensorflow as tf
import numpy as np

from tfdet.core.metric import (MeanAveragePrecision as MeanAveragePrecisionMetric,
                               CoCoMeanAveragePrecision as CoCoMeanAveragePrecisionMetric,
                               MeanIoU as MeanIoUMetric)

class MeanAveragePrecision(tf.keras.callbacks.Callback):
    """
    scale_range = None > area_range = [[None, None]] # 0~INF (all scale)
    scale_range = [96] > area_range = [[None, 96^2], [96^2, None]] # 0~96^2, 96^2~INF
    scale_range = [32, 96] > area_range = [[None, 32^2], [32^2, 96^2], [96^2, None]] # 0~32^2, 32^2~96^2, 96^2~INF
    scale_range = [None, 32, 96] > area_range = [[None, None], [None, 32^2], [32^2, 96^2], [96^2, None]] #0~INF, 0~32^2, 32^2~96^2, 96^2~INF
    scale_range = [32, None, 96] > area_range = [[None, 32^2], [32^2, None], [None, 96^2], [96^2, None]] #0~32^2, 32^2~INF, 0~96^2, 96^2~INF
    """
    def __init__(self, data, iou_threshold = 0.5, score_threshold = 0.05, scale_range = None, mode = "normal", e = 1e-12, postfix = False, label = None, dtype = np.float32, verbose = True, name = "mean_average_precision", **kwargs):
        super(MeanAveragePrecision, self).__init__(**kwargs)
        self.data = data
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.scale_range = scale_range
        self.mode = mode
        self.e = e
        self.postfix = postfix
        self.label = label
        self.dtype = dtype
        self.verbose = verbose
        self.name = name
        
        self.metric = MeanAveragePrecisionMetric(iou_threshold = self.iou_threshold, score_threshold = self.score_threshold, scale_range = self.scale_range, mode = self.mode, e = self.e, postfix = self.postfix, label = self.label, dtype = self.dtype)
    
    def evaluate(self):
        self.metric.reset()
        input_key = [inp.name for inp in self.model.inputs]
        input_cnt = len(input_key)
        iterator = iter(self.data if isinstance(self.data, tf.data.Dataset) else [self.data])
        while True:
            try:
                data = next(iterator)
                if not isinstance(data, dict):
                    x = data[:input_cnt]
                    y_true, bbox_true = list(data[input_cnt:])[:2]
                else:
                    x = [data[k] for k in input_key if k in data]
                    y_true, bbox_true = data["y_true"], data["bbox_true"]

                y_pred, bbox_pred = self.model.predict(x, verbose = 0)[:2]
                self.metric.add(y_true, bbox_true, y_pred, bbox_pred)
                del data, x, y_true, bbox_true, y_pred, bbox_pred
            except:
                break
        return self.metric.evaluate()
    
    def on_epoch_begin(self, epoch, logs = None):
        self.metric.reset()
    
    def on_epoch_end(self, epoch, logs = {}):
        if self.postfix:
            min_iou_threshold = str(np.min(self.iou_threshold)).replace("0.", ".")
            max_iou_threshold = str(np.max(self.iou_threshold)).replace("0.", ".")
            postfix = "@{0}".format(min_iou_threshold)
            if min_iou_threshold != max_iou_threshold:
                postfix = "{0}:{1}".format(postfix, max_iou_threshold)
        else:
            postfix = ""
        
        mean_average_precision = self.evaluate()
        mean_average_precision = [mean_average_precision] if np.ndim(mean_average_precision) == 0 else mean_average_precision

        post = lambda x:int(x)
        if self.scale_range is not None:
            scale_range = [self.scale_range] if np.ndim(self.scale_range) == 0 else self.scale_range
            scale_range = [scale for scale in scale_range if scale is not None]
            if np.issubdtype(np.array(scale_range).dtype, np.floating):
                post = lambda x:float(x)
        area_range = [[str(post(min_area**(1/2) if min_area is not None else 0)), str(post(max_area**(1/2))) if max_area is not None else "INF"] for (min_area, max_area) in self.metric.area_range]
        all_range_flag = [float(min_area) == 0 and max_area == "INF" for (min_area, max_area) in area_range]
        scale_range_flag = np.invert(all_range_flag)
        if 0 < np.sum(all_range_flag):
            i = np.argwhere(all_range_flag)[:, 0][0]
            logs["{0}{1}".format(self.name, postfix)] = mean_average_precision[i]
        for i in np.argwhere(scale_range_flag)[:, 0]:
            logs["{0}{1}[{2}:{3}]".format(self.name, postfix, *area_range[i])] = mean_average_precision[i]
        
        if self.verbose:
            text = self.metric.summary_text
            if 0 < len(text):
                print("\n{0}".format(text))
        
class CoCoMeanAveragePrecision(tf.keras.callbacks.Callback):
    """
    scale_range = None > area_range = [[None, None]] # 0~INF (all scale)
    scale_range = [96] > area_range = [[None, 96^2], [96^2, None]] # 0~96^2, 96^2~INF
    scale_range = [32, 96] > area_range = [[None, 32^2], [32^2, 96^2], [96^2, None]] # 0~32^2, 32^2~96^2, 96^2~INF
    scale_range = [None, 32, 96] > area_range = [[None, None], [None, 32^2], [32^2, 96^2], [96^2, None]] #0~INF, 0~32^2, 32^2~96^2, 96^2~INF
    scale_range = [32, None, 96] > area_range = [[None, 32^2], [32^2, None], [None, 96^2], [96^2, None]] #0~32^2, 32^2~INF, 0~96^2, 96^2~INF
    """
    def __init__(self, data, iou_threshold = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], score_threshold = 0.05, scale_range = None, mode = "normal", e = 1e-12, postfix = False, label = None, dtype = np.float32, verbose = True, name = "mean_average_precision", **kwargs):
        super(CoCoMeanAveragePrecision, self).__init__(**kwargs)
        self.data = data
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.scale_range = scale_range
        self.mode = mode
        self.e = e
        self.postfix = postfix
        self.label = label
        self.dtype = dtype
        self.verbose = verbose
        self.name = name
        
        self.metric = CoCoMeanAveragePrecisionMetric(iou_threshold = self.iou_threshold, score_threshold = self.score_threshold, scale_range = self.scale_range, mode = self.mode, e = self.e, postfix = self.postfix, label = self.label, dtype = self.dtype)
    
    def evaluate(self):
        self.metric.reset()
        input_key = [inp.name for inp in self.model.inputs]
        input_cnt = len(input_key)
        iterator = iter(self.data if isinstance(self.data, tf.data.Dataset) else [self.data])
        while True:
            try:
                data = next(iterator)
                if not isinstance(data, dict):
                    x = data[:input_cnt]
                    y_true, bbox_true = list(data[input_cnt:])[:2]
                else:
                    x = [data[k] for k in input_key if k in data]
                    y_true, bbox_true = data["y_true"], data["bbox_true"]

                y_pred, bbox_pred = self.model.predict(x, verbose = 0)[:2]
                self.metric.add(y_true, bbox_true, y_pred, bbox_pred)
                del data, x, y_true, bbox_true, y_pred, bbox_pred
            except:
                break
        return self.metric.evaluate()
    
    def on_epoch_begin(self, epoch, logs = None):
        self.metric.reset()
    
    def on_epoch_end(self, epoch, logs = {}):
        if self.postfix:
            min_iou_threshold = str(np.min(self.iou_threshold)).replace("0.", ".")
            max_iou_threshold = str(np.max(self.iou_threshold)).replace("0.", ".")
            postfix = "@{0}".format(min_iou_threshold)
            if min_iou_threshold != max_iou_threshold:
                postfix = "{0}:{1}".format(postfix, max_iou_threshold)
        else:
            postfix = ""
        
        mean_average_precision = self.evaluate()
        mean_average_precision = [mean_average_precision] if np.ndim(mean_average_precision) == 0 else mean_average_precision
        mean_average_precision_50 = mean_average_precision_75 = 0.
        if self.metric.sub_metric.num_pred is not None and 0 < np.sum(self.metric.sub_metric.num_pred):
            average_precision = self.metric.sub_metric.evaluate(reduce = False)
            true_flag = self.metric.sub_metric.num_true[..., 0] != 0
            average_precision_50, average_precision_75 = average_precision[..., 0], average_precision[..., 1]
            mean_average_precision_50 = np.mean(average_precision_50[true_flag])
            mean_average_precision_75 = np.mean(average_precision_75[true_flag])

        post = lambda x:int(x)
        if self.scale_range is not None:
            scale_range = [self.scale_range] if np.ndim(self.scale_range) == 0 else self.scale_range
            scale_range = [scale for scale in scale_range if scale is not None]
            if np.issubdtype(np.array(scale_range).dtype, np.floating):
                post = lambda x:float(x)
        area_range = [[str(post(min_area**(1/2) if min_area is not None else 0)), str(post(max_area**(1/2))) if max_area is not None else "INF"] for (min_area, max_area) in self.metric.metric.area_range]
        all_range_flag = [float(min_area) == 0 and max_area == "INF" for (min_area, max_area) in area_range]
        scale_range_flag = np.invert(all_range_flag)
        if 0 < np.sum(all_range_flag):
            i = np.argwhere(all_range_flag)[:, 0][0]
            logs["{0}{1}".format(self.name, postfix)] = mean_average_precision[i]
        logs["{0}@.5".format(self.name)] = mean_average_precision_50
        logs["{0}@.75".format(self.name)] = mean_average_precision_75
        for i in np.argwhere(scale_range_flag)[:, 0]:
            logs["{0}{1}[{2}:{3}]".format(self.name, postfix, *area_range[i])] = mean_average_precision[i]
        
        if self.verbose:
            text = self.metric.summary_text
            if 0 < len(text):
                print("\n{0}".format(text))
                
class MeanIoU(tf.keras.callbacks.Callback):
    def __init__(self, data, beta = 1, e = 1e-12, label = None, dtype = np.float32, verbose = True, name = "mean_iou", **kwargs):
        super(MeanIoU, self).__init__(**kwargs)
        self.data = data
        self.beta = beta
        self.e = e
        self.label = label
        self.dtype = dtype
        self.verbose = verbose
        self.name = name
        
        self.metric = MeanIoUMetric(beta = self.beta, e = self.e, label = self.label, dtype = self.dtype)
    
    def evaluate(self):
        self.metric.reset()
        input_key = [inp.name for inp in self.model.inputs]
        input_cnt = len(input_key)
        iterator = iter(self.data if isinstance(self.data, tf.data.Dataset) else [self.data])
        while True:
            try:
                data = next(iterator)
                if not isinstance(data, dict):
                    x = data[:input_cnt]
                    mask_true = list(data[input_cnt:])[0]
                else:
                    x = [data[k] for k in input_key if k in data]
                    mask_true = data["mask_true"] if "mask_true" in data else data["y_true"]

                mask_pred = self.model.predict(x, verbose = 0)
                if isinstance(mask_pred, tuple):
                    mask_pred = mask_pred[0]
                self.metric.add(mask_true, mask_pred)
                del data, x, mask_true, mask_pred
            except:
                break
        return self.metric.evaluate()
    
    def on_epoch_begin(self, epoch, logs = None):
        self.metric.reset()
    
    def on_epoch_end(self, epoch, logs = {}):
        mean_iou = self.evaluate()
        mean_accuracy = mean_dice = mean_f1 = 0.
        if self.metric.area_true is not None:
            mean_accuracy = self.metric.mean_accuracy
            mean_dice = self.metric.mean_dice
            mean_f1 = self.metric.mean_f1
        logs["mean_accuracy"] = mean_accuracy
        logs[self.name] = mean_iou
        logs["mean_dice"] = mean_dice
        logs["mean_f1"] = mean_f1
            
        if self.verbose:
            text = self.metric.summary_text
            if 0 < len(text):
                print("\n{0}".format(text))
