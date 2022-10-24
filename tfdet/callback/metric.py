import tensorflow as tf
import numpy as np

from tfdet.core.metric import (MeanAveragePrecision as MeanAveragePrecisionMetric,
                               CoCoMeanAveragePrecision as CoCoMeanAveragePrecisionMetric,
                               MeanIoU as MeanIoUMetric)

class MeanAveragePrecision(tf.keras.callbacks.Callback):
    def __init__(self, data, iou_threshold = 0.5, score_threshold = 0.05, mode = "normal", e = 1e-12, label = None, verbose = True, name = "mean_average_precision", **kwargs):
        super(MeanAveragePrecision, self).__init__(**kwargs)
        self.data = data
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.mode = mode
        self.e = e
        self.label = label
        self.verbose = verbose
        self.name = name
        
        self.metric = MeanAveragePrecisionMetric(iou_threshold = self.iou_threshold, score_threshold = self.score_threshold, mode = self.mode, e = self.e, label = self.label)
    
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
            except:
                break
        return self.metric.evaluate()
    
    def on_epoch_end(self, epoch, logs = {}):
        logs[self.name] = self.evaluate()
        
        if self.verbose:
            text = self.metric.summary_text
            if 0 < len(text):
                print("\n{0}".format(text))
        
class CoCoMeanAveragePrecision(tf.keras.callbacks.Callback):
    def __init__(self, data, iou_threshold = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.], score_threshold = 0.05, mode = "normal", e = 1e-12, label = None, verbose = True, name = "mean_average_precision", **kwargs):
        super(CoCoMeanAveragePrecision, self).__init__(**kwargs)
        self.data = data
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.mode = mode
        self.e = e
        self.label = label
        self.verbose = verbose
        self.name = name
        
        self.metric = CoCoMeanAveragePrecisionMetric(iou_threshold = self.iou_threshold, score_threshold = self.score_threshold, mode = self.mode, e = self.e, label = self.label)
    
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
            except:
                break
        return self.metric.evaluate()
    
    def on_epoch_end(self, epoch, logs = {}):
        mean_average_precision = self.evaluate()
        mean_average_precision_50 = mean_average_precision_75 = 0.
        if self.metric.num_true is not None:
            average_precision = self.metric.sub_metric.evaluate(reduce = False)
            average_precision_50, average_precision_75 = average_precision[..., 0], average_precision[..., 1]
            mean_average_precision_50 = np.mean(average_precision_50[self.metric.num_true[..., 0] != 0]).item()
            mean_average_precision_75 = np.mean(average_precision_75[self.metric.num_true[..., 0] != 0]).item()
        logs["{0}_50".format(self.name)] = mean_average_precision_50
        logs["{0}_75".format(self.name)] = mean_average_precision_75
        logs[self.name] = mean_average_precision
        
        if self.verbose:
            text = self.metric.summary_text
            if 0 < len(text):
                print("\n{0}".format(text))
                
class MeanIoU(tf.keras.callbacks.Callback):
    def __init__(self, data, beta = 1, e = 1e-12, label = None, verbose = True, name = "mean_iou", **kwargs):
        super(MeanIoU, self).__init__(**kwargs)
        self.data = data
        self.beta = beta
        self.e = e
        self.label = label
        self.verbose = verbose
        self.name = name
        
        self.metric = MeanIoUMetric(beta = self.beta, e = self.e, label = self.label)
    
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
                    mask_true = list(data[input_cnt:])[:1]
                else:
                    x = [data[k] for k in input_key if k in data]
                    mask_true = data["mask_true"] if "mask_true" in data else data["y_true"]

                mask_pred = self.model.predict(x, verbose = 0)
                if isinstance(mask_pred, tuple):
                    mask_pred = mask_pred[0]
                self.metric.add(mask_true, mask_pred)
            except:
                break
        return self.metric.evaluate()
    
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