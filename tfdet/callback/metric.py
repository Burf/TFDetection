import tensorflow as tf
import numpy as np

from tfdet.core.metric import (MeanAveragePrecision as MeanAveragePrecisionMetric,
                               CoCoMeanAveragePrecision as CoCoMeanAveragePrecisionMetric)
from tfdet.core.util import metric2text

class MeanAveragePrecision(tf.keras.callbacks.Callback):
    def __init__(self, data, iou_threshold = 0.5, score_threshold = 0.05, mode = "normal", eval_mode = "area", e = 1e-12, label = None, verbose = True, name = "mean_average_precision", **kwargs):
        super(MeanAveragePrecision, self).__init__(**kwargs)
        self.data = data
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.mode = mode
        self.eval_mode = eval_mode
        self.e = e
        self.label = label
        self.verbose = verbose
        self.name = name
        
        self.metric = MeanAveragePrecisionMetric(iou_threshold = self.iou_threshold, score_threshold = self.score_threshold, mode = self.mode, e = self.e)
    
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
        return self.metric.evaluate(mode = self.eval_mode)
    
    @property
    def summary(self):
        text = ""
        if self.metric.num_true is not None:
            precision, recall, average_precision = self.metric.evaluate(reduce = True, return_summary = True, mode = self.eval_mode)
            num_true = self.metric.num_true[..., 0].astype(int)
            num_pred = self.metric.num_pred[..., 0].astype(int)
            info = {"num_true":num_true, "num_pred":num_pred, "precision":precision, "recall":recall, "average_precision":average_precision}
            
            precision = np.mean(precision[num_true != 0]).item()
            recall = np.mean(recall[num_true != 0]).item()
            average_precision = np.mean(average_precision[num_true != 0]).item()
            num_true = np.sum(num_true)
            num_pred = np.sum(num_pred)
            summary = [num_true, num_pred, precision, recall, average_precision]
            
            text = metric2text(info, summary = summary, label = self.label)
        return text

    def on_epoch_end(self, epoch, logs = {}):
        logs[self.name] = self.evaluate()
        
        if self.verbose:
            text = self.summary
            if 0 < len(text):
                print("\n{0}".format(text))
        
class CoCoMeanAveragePrecision(tf.keras.callbacks.Callback):
    def __init__(self, data, iou_threshold = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.], score_threshold = 0.05, mode = "normal", eval_mode = "area", e = 1e-12, label = None, verbose = True, name = "mean_average_precision", **kwargs):
        super(CoCoMeanAveragePrecision, self).__init__(**kwargs)
        self.data = data
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.mode = mode
        self.eval_mode = eval_mode
        self.e = e
        self.label = label
        self.verbose = verbose
        self.name = name
        
        self.metric = CoCoMeanAveragePrecisionMetric(iou_threshold = self.iou_threshold, score_threshold = self.score_threshold, mode = self.mode, e = self.e)
    
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
        return self.metric.evaluate(mode = self.eval_mode)
    
    @property
    def summary(self):
        text = ""
        if self.metric.num_true is not None:
            precision, recall, average_precision = self.metric.evaluate(reduce = True, return_summary = True, mode = self.eval_mode)
            sub_precision, sub_recall, sub_average_precision = self.metric.sub_metric.evaluate(reduce = False, return_summary = True, mode = self.eval_mode)
            precision_50, recall_50, average_precision_50 = sub_precision[..., 0], sub_recall[..., 0], sub_average_precision[..., 0]
            precision_75, recall_75, average_precision_75 = sub_precision[..., 1], sub_recall[..., 1], sub_average_precision[..., 1]
            num_true = self.metric.num_true[..., 0].astype(int)
            num_pred = self.metric.num_pred[..., 0].astype(int)
            info = {"num_true":num_true, "num_pred":num_pred,
                    "precision":precision, "precision_50":precision_50, "precision_75":precision_75,
                    "recall":recall, "recall_50":recall_50, "recall_75":recall_75,
                    "average_precision":average_precision, "average_precision_50":average_precision_50, "average_precision_75":average_precision_75}
            
            precision = np.mean(precision[num_true != 0]).item()
            precision_50 = np.mean(precision_50[num_true != 0]).item()
            precision_75 = np.mean(precision_75[num_true != 0]).item()
            recall = np.mean(recall[num_true != 0]).item()
            recall_50 = np.mean(recall_50[num_true != 0]).item()
            recall_75 = np.mean(recall_75[num_true != 0]).item()
            average_precision = np.mean(average_precision[num_true != 0]).item()
            average_precision_50 = np.mean(average_precision_50[num_true != 0]).item()
            average_precision_75 = np.mean(average_precision_75[num_true != 0]).item()
            num_true = np.sum(num_true)
            num_pred = np.sum(num_pred)
            summary = [num_true, num_pred, precision, precision_50, precision_75,recall, recall_50, recall_75, average_precision, average_precision_50, average_precision_75]
            
            text = metric2text(info, summary = summary, label = self.label)
        return text

    def on_epoch_end(self, epoch, logs = {}):
        logs[self.name] = self.evaluate()
        if self.metric.num_true is not None:
            average_precision = self.metric.sub_metric.evaluate(reduce = False, mode = self.eval_mode)
            average_precision_50, average_precision_75 = average_precision[..., 0], average_precision[..., 1]
            logs["{0}_50".format(self.name)] = np.mean(average_precision_50[self.metric.num_true[..., 0] != 0]).item()
            logs["{0}_75".format(self.name)] = np.mean(average_precision_50[self.metric.num_true[..., 0] != 0]).item()
            
        if self.verbose:
            text = self.summary
            if 0 < len(text):
                print("\n{0}".format(text))