import tensorflow as tf
import numpy as np

from ..bbox import overlap_bbox_numpy as overlap_bbox

class MeanAveragePrecision:
    def __init__(self, iou_threshold = 0.5, score_threshold = 0.05, mode = "normal", e = 1e-12):
        """
        run = class_instance(*args)
        batch run = add(*batch_args) -> evaluate()
        """
        self.iou_threshold = np.array([iou_threshold]) if np.ndim(iou_threshold) == 0 else np.array(iou_threshold)
        self.score_threshold = score_threshold
        self.mode = mode
        self.e = e
        
        self.reset()
        
    def reset(self):
        self.tp = []
        self.fp = []
        self.score_pred = []
        self.num_true = None
        self.num_pred = None
    
    @property
    def precision_n_recall(self):
        return self.evaluate(reduce = False, return_precision_n_recall = True)
    
    @property
    def precision(self):
        return self.precision_n_recall[0]
    
    @property
    def recall(self):
        return self.precision_n_recall[1]
    
    @property
    def f1(self):
        precision, recall = self.precision_n_recall
        return 2 * (precision * recall) / np.maximum(precision + recall, self.e)
    
    @property
    def threshold(self):
        if self.num_true is not None:
            #sort_indices = np.argsort(-self.iou_threshold)
            #return np.where(self.num_true[..., 0] != 0, self.iou_threshold[sort_indices][np.argmax(self.f1[:, sort_indices], axis = -1)], 0)
            return np.where(self.num_true[..., 0] != 0, self.iou_threshold[np.argmax(self.f1, axis = -1)], 0)
    
    @property
    def average_precision(self):
        return self.evaluate(reduce = False)
    
    @property
    def mean_average_precision(self):
        return self.evaluate()
    
    @property
    def summary(self):
        return self.evaluate(reduce = False, return_summary = True)
    
    def __call__(self, y_true, bbox_true, y_pred, bbox_pred, reduce = True, return_precision_n_recall = False, mode = "area", return_summary = False, reset = True):
        """
        y_true = label #(batch_size, padded_num_true, 1 or n_class)
        bbox_true = [[x1, y1, x2, y2], ...] #(batch_size, padded_num_true, bbox)
        y_pred = classifier logit #(batch_size, num_proposals, num_class)
        bbox_pred = classifier regress #(batch_size, num_proposals, delta)
        """
        if reset:
            self.reset()
        self.add(y_true, bbox_true, y_pred, bbox_pred)
        return self.evaluate(reduce = reduce, return_precision_n_recall = return_precision_n_recall, mode = mode, return_summary = return_summary)
    
    def add(self, y_true, bbox_true, y_pred, bbox_pred):
        """
        y_true = label #(padded_num_true, 1 or n_class) or (batch_size, padded_num_true, 1 or n_class)
        bbox_true = [[x1, y1, x2, y2], ...] #(padded_num_true, bbox) or (batch_size, padded_num_true, bbox)
        y_pred = classifier logit #(num_proposals, num_class) or (batch_size, num_proposals, num_class)
        bbox_pred = classifier regress #(num_proposals, delta) or (batch_size, num_proposals, delta)
        """
        if np.ndim(y_true) == 3:
            for index in range(len(y_true)):
                self.add(y_true[index], bbox_true[index], y_pred[index], bbox_pred[index])
        else:
            n_class = np.shape(y_pred)[-1]
            if self.num_true is None:
                self.num_true = np.zeros((n_class, 1))
                self.num_pred = np.zeros((n_class, 1))
            
            if 1 < np.shape(y_true)[-1]:
                y_true = np.expand_dims(np.argmax(y_true, axis = -1), axis = -1)
            valid_indices = np.where(np.max(0 < bbox_true, axis = -1))
            y_true = np.array(y_true)[valid_indices]
            bbox_true = np.array(bbox_true)[valid_indices]
            valid_indices = np.where(np.max(0 < bbox_pred, axis = -1))
            y_pred = np.array(y_pred)[valid_indices]
            bbox_pred = np.array(bbox_pred)[valid_indices]
            
            label = np.argmax(y_pred, axis = -1)
            score = np.max(y_pred, axis = -1)
            sort_indices = np.argsort(-score)
            label = label[sort_indices]
            score = score[sort_indices]
            bbox_pred = bbox_pred[sort_indices]
            score_flag = self.score_threshold <= score
            
            tp = np.zeros((n_class, len(y_pred), len(self.iou_threshold)))
            fp = np.zeros((n_class, len(y_pred), len(self.iou_threshold)))
            for cls in range(n_class):
                true_flag = y_true[..., 0] == cls
                pred_flag = np.logical_and(label == cls, score_flag)
                true_count = np.sum(true_flag)
                pred_count = np.sum(pred_flag)
                if 0 < pred_count:
                    if true_count == 0:
                        fp[cls] = 1
                    else:
                        overlaps = overlap_bbox(bbox_pred[pred_flag], bbox_true[true_flag], mode = self.mode) #(P, T)
                        max_iou = np.max(overlaps, axis = 1) #(P,)
                        argmax_iou = np.argmax(overlaps, axis = 1) #(P,)
                        covered = np.zeros((true_count, len(self.iou_threshold)), dtype = bool)
                        for i, iou_threshold in enumerate(self.iou_threshold):
                            for p in range(pred_count):
                                if iou_threshold <= max_iou[p]:
                                    t = argmax_iou[p]
                                    if not covered[t, i]:
                                        covered[t, i] = True
                                        tp[cls, p, i] = 1
                                    else:
                                        fp[cls, p, i] = 1
                                else:
                                    fp[cls, p, i] = 1
                    self.num_pred[cls] += pred_count
                if 0 < true_count:
                    self.num_true[cls] += true_count
            self.tp.append(tp)
            self.fp.append(fp)
            self.score_pred.append(score)
    
    def evaluate(self, reduce = True, return_precision_n_recall = False, mode = "area", return_summary = False):
        if self.num_true is not None:
            if mode not in ["area", "11points"]:
                raise ValueError("unknown mode '{0}'".format(mode))

            sort_indices = np.argsort(-np.hstack(self.score_pred))
            tp = np.hstack(self.tp)[:, sort_indices]
            fp = np.hstack(self.fp)[:, sort_indices]
            tp = np.cumsum(tp, axis = 1)
            fp = np.cumsum(fp, axis = 1)
            precision = tp / np.maximum(tp + fp, self.e)
            recall = tp / np.maximum(np.expand_dims(self.num_true, axis = 1), self.e)
            precision = np.where(self.num_true[..., None] != 0, precision, 0)
            recall = np.where(self.num_true[..., None] != 0, recall, 0)
            
            if return_precision_n_recall:
                precision = precision[:, -1]
                recall = recall[:, -1]
                if reduce:
                    precision = np.mean(precision, axis = 1)
                    recall = np.mean(recall, axis = 1)
                return [precision, recall]
            
            n_class = np.shape(self.tp[0])[0]
            average_precision = np.zeros((n_class, len(self.iou_threshold)))
            if mode == "area":
                zeros = np.zeros((n_class, 1, len(self.iou_threshold)))
                ones = np.ones((n_class, 1, len(self.iou_threshold)))
                mpre = np.hstack([zeros, precision, zeros])
                mrec = np.hstack([zeros, recall, ones])
                for i in range(np.shape(mpre)[1] - 1, 0, -1):
                    mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
                for cls in range(n_class):
                    for th in range(len(self.iou_threshold)):
                        indices = np.where(mrec[cls, 1:, th] != mrec[cls, :-1, th])[0]
                        average_precision[cls, th] = np.sum((mrec[cls, indices + 1, th] - mrec[cls, indices, th]) * mpre[cls, indices + 1, th])
            elif mode in "11points":
                zeros = np.zeros_like(precision)
                for recall_threshold in np.linspace(0., 1., 11):
                    pre = np.where(recall_threshold <= recall, precision, zeros)
                    average_precision += np.max(pre, axis = 1)
                average_precision /= 11
                
            if return_summary:
                precision = precision[:, -1]
                recall = recall[:, -1]
                if reduce:
                    precision = np.mean(precision, axis = 1)
                    recall = np.mean(recall, axis = 1)
                    average_precision = np.mean(average_precision, axis = 1)
                return [precision, recall, average_precision]
            if reduce:
                average_precision = np.mean(average_precision[self.num_true[..., 0] != 0]).item()
            return average_precision
        else:
            if return_precision_n_recall:
                return [[0.] * len(self.iou_threshold), [0.] * len(self.iou_threshold)]
            elif return_summary:
                return [[0.] * len(self.iou_threshold), [0.] * len(self.iou_threshold), [0.] * len(self.iou_threshold)]
            else:
                return 0.
    
class CoCoMeanAveragePrecision:
    def __init__(self, iou_threshold = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.], score_threshold = 0.05, mode = "normal", e = 1e-12):
        self.iou_threshold = np.array([iou_threshold]) if np.ndim(iou_threshold) == 0 else np.array(iou_threshold)
        self.score_threshold = score_threshold
        self.mode = mode
        self.e = e
        
        self.reset()
        
    def reset(self):
        self.metric = MeanAveragePrecision(iou_threshold = self.iou_threshold, score_threshold = self.score_threshold, mode = self.mode, e = self.e)
        self.sub_metric = MeanAveragePrecision(iou_threshold = [0.5, 0.75], score_threshold = self.score_threshold, mode = self.mode, e = self.e)
    
    @property
    def num_true(self):
        return self.metric.num_true
    
    @property
    def num_pred(self):
        return self.metric.num_pred

    @property
    def precision_n_recall(self):
        return self.metric.precision_n_recall
    
    @property
    def precision(self):
        return self.precision_n_recall[0]
    
    @property
    def recall(self):
        return self.precision_n_recall[1]
    
    @property
    def average_precision(self):
        return self.metric.average_precision
    
    @property
    def mean_average_precision(self):
        return self.metric.mean_average_precision
    
    @property
    def f1(self):
        return self.metric.f1
    
    @property
    def threshold(self):
        return self.metric.threshold
    
    @property
    def summary(self):
        return self.metric.summary

    @property
    def precision_n_recall_50(self):
        return self.sub_metric.precision_n_recall[..., :1]
    
    @property
    def precision_50(self):
        return self.precision_n_recall_50[0]
    
    @property
    def recall_50(self):
        return self.precision_n_recall_50[1]
    
    @property
    def f1_50(self):
        return self.sub_metric.f1[..., :1]
    
    @property
    def average_precision_50(self):
        return self.sub_metric.average_precision[..., :1]
    
    @property
    def mean_average_precision_50(self):
        return np.mean(self.average_precision_50[self.num_true[..., 0] != 0]).item()
    
    @property
    def summary_50(self):
        return [v[..., :1] for v in self.sub_metric.evaluate(reduce = False, return_summary = True)]

    @property
    def precision_n_recall_75(self):
        return self.sub_metric.precision_n_recall[..., 1:]
    
    @property
    def precision_75(self):
        return self.precision_n_recall_75[0]
    
    @property
    def recall_75(self):
        return self.precision_n_recall_75[1]
    
    @property
    def f1_75(self):
        return self.sub_metric.f1[..., 1:]
    
    @property
    def average_precision_75(self):
        return self.sub_metric.average_precision[..., 1:]
    
    @property
    def mean_average_precision_75(self):
        return np.mean(self.average_precision_75[self.num_true[..., 0] != 0]).item()
    
    @property
    def summary_75(self):
        return [v[..., 1:] for v in self.sub_metric.evaluate(reduce = False, return_summary = True)]
    
    def __call__(self, y_true, bbox_true, y_pred, bbox_pred, reduce = True, return_precision_n_recall = False, mode = "area", return_summary = False, reset = True):
        """
        y_true = label #(batch_size, padded_num_true, 1 or n_class)
        bbox_true = [[x1, y1, x2, y2], ...] #(batch_size, padded_num_true, bbox)
        y_pred = classifier logit #(batch_size, num_proposals, num_class)
        bbox_pred = classifier regress #(batch_size, num_proposals, delta)
        """
        if reset:
            self.reset()
        self.add(y_true, bbox_true, y_pred, bbox_pred)
        return self.evaluate(reduce = reduce, return_precision_n_recall = return_precision_n_recall, mode = mode, return_summary = return_summary)
    
    def add(self, y_true, bbox_true, y_pred, bbox_pred):
        """
        y_true = label #(padded_num_true, 1 or n_class) or (batch_size, padded_num_true, 1 or n_class)
        bbox_true = [[x1, y1, x2, y2], ...] #(padded_num_true, bbox) or (batch_size, padded_num_true, bbox)
        y_pred = classifier logit #(num_proposals, num_class) or (batch_size, num_proposals, num_class)
        bbox_pred = classifier regress #(num_proposals, delta) or (batch_size, num_proposals, delta)
        """
        self.metric.add(y_true, bbox_true, y_pred, bbox_pred)
        self.sub_metric.add(y_true, bbox_true, y_pred, bbox_pred)
                
    def evaluate(self, reduce = True, return_precision_n_recall = False, mode = "area", return_summary = False):
        return self.metric.evaluate(reduce = reduce, return_precision_n_recall = return_precision_n_recall, mode = mode, return_summary = return_summary)