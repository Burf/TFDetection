import tensorflow as tf
import numpy as np

from tfdet.core.metric import MeanAveragePrecision as MeanAveragePrecisionMetric

class MeanAveragePrecision(tf.keras.callbacks.Callback):
    def __init__(self, data, n_class, threshold = 0.5, score_threshold = 0.05, r = 11, interpolate = True, mode = "normal", name = "mean_average_precision", **kwargs):
        super(MeanAveragePrecision, self).__init__(**kwargs)
        self.data = data
        self.n_class = n_class
        self.threshold = threshold
        self.score_threshold = score_threshold
        self.r = r
        self.interpolate = interpolate
        self.mode = mode
        self.name = name
        
        self.metric = MeanAveragePrecisionMetric(n_class = self.n_class, threshold = self.threshold, score_threshold = self.score_threshold, r = self.r, mode = self.mode)
    
    def reset(self):
        self.metric.reset()
    
    def evaluate(self):
        input_key = [inp.name for inp in self.model.inputs]
        input_cnt = len(input_key)
        iterator = iter(self.data)
        while True:
            try:
                data = next(iterator)
                if not isinstance(data, dict):
                    x = data[:input_cnt]
                    y_true, bbox_true = list(data[input_cnt:])[:2]
                else:
                    x = [data[k] for k in input_key if k in data]
                    y_true, bbox_true = data["y_true"], data["bbox_true"]

                y_pred, bbox_pred = self.model.predict(x, verbose = 0)
                
                self.metric.add(y_true, bbox_true, y_pred, bbox_pred)
            except:
                break
        return self.metric.evaluate(interpolate = self.interpolate)

    def on_epoch_end(self, epoch, logs = {}):
        self.reset()
        logs[self.name] = self.evaluate()
        
    def on_test_end(self, logs = {}):
        self.reset()
        logs[self.name] = self.evaluate()
    
    def on_predict_end(self, logs = {}):
        self.reset()
        logs[self.name] = self.evaluate()