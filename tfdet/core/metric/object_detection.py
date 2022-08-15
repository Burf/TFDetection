import tensorflow as tf
import numpy as np

from ..bbox import overlap_bbox 

class MeanAveragePrecision:
    def __init__(self, n_class, threshold = 0.5, score_threshold = 0.05, r = 11, mode = "normal"):
        """
        run = class_instance(*args)
        batch run = add(*batch_args) -> evaluate()
        """
        self.n_class = n_class
        self.threshold = threshold
        self.score_threshold = score_threshold
        self.r = r
        self.mode = mode
        
        self.r_threshold = np.linspace(0., 1., r)
        self.reset()
        
    def reset(self):
        self.tp = np.zeros([self.n_class, self.r])
        self.fp = np.zeros([self.n_class, self.r])
        self.fn = np.zeros([self.n_class, self.r])
    
    @property
    def precision(self):
        return self.evaluate(interpolate = True, return_precision_n_recall = True)[0]
    
    @property
    def recall(self):
        return self.evaluate(interpolate = True, return_precision_n_recall = True)[1]
    
    @property
    def precision_n_recall(self):
        return self.evaluate(interpolate = True, return_precision_n_recall = True)
    
    @property
    def average_precision(self):
        return self.evaluate(interpolate = True, reduce = False)
    
    @property
    def mean_average_precision(self):
        return self.evaluate(interpolate = True, reduce = True)
    
    @property
    def f1(self):
        precision, recall = self.evaluate(interpolate = True, return_precision_n_recall = True)
        pr = precision + recall
        return 2 * (precision * recall) / np.where(pr == 0, 1e-12, pr)
        
    def __call__(self, y_true, bbox_true, y_pred, bbox_pred, interpolate = True, reduce = True, return_precision_n_recall = False):
        """
        y_true = label #(batch_size, padded_num_true, 1 or n_class)
        bbox_true = [[x1, y1, x2, y2], ...] #(batch_size, padded_num_true, bbox)
        y_pred = classifier logit #(batch_size, num_proposals, num_class)
        bbox_pred = classifier regress #(batch_size, num_proposals, delta)
        """
        self.reset()
        self.add(y_true, bbox_true, y_pred, bbox_pred)
        return self.evaluate(interpolate = interpolate, reduce = reduce, return_precision_n_recall = return_precision_n_recall)
    
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
            tile_score = np.tile(np.expand_dims(score, axis = 0), [self.r, 1])
            overlaps = self.threshold <= np.transpose(overlap_bbox(bbox_true.astype(np.float32), bbox_pred.astype(np.float32), self.mode)) #(P, T)
            for c in range(self.n_class):
                true_flag = y_true[..., 0] == c
                true_count = np.sum(true_flag)
                pred_flag = np.logical_and(label == c, self.score_threshold <= score)
                pred_count = np.sum(pred_flag)
                if pred_count == 0:
                    if true_count == 0:
                        continue
                    else:
                        self.fn[c] += true_count
                else:
                    tile_pred_flag = np.tile(np.expand_dims(pred_flag, axis = 0), [self.r, 1])
                    r_flag = np.logical_and(tile_pred_flag, np.expand_dims(self.r_threshold, axis = -1) <= tile_score)
                    r_count = np.sum(r_flag, axis = -1)

                    if true_count == 0:
                        self.fp[c] += r_count
                    else:
                        mask = overlaps[:, true_flag]
                        tp_count = [np.sum(np.any(mask[flag], axis = 0)) for flag in r_flag]
                        self.tp[c] += tp_count
                        self.fp[c] += (r_count - tp_count)
                        self.fn[c] += (true_count - tp_count)
    
    def evaluate(self, interpolate = True, reduce = True, return_precision_n_recall = False):
        tp_fp = self.tp + self.fp
        tp_fn = self.tp + self.fn
        precision = np.where(tp_fp == 0, np.where(tp_fn == 0, 1, 0), self.tp / np.where(tp_fp == 0, 1, tp_fp))
        recall = np.where(tp_fn == 0, 1, self.tp / np.where(tp_fn == 0, 1, tp_fn))
        if interpolate:
            for index in range(1, self.r):
                p = np.max(precision[..., :index + 1], axis = 1)
                precision[:, index] = p
        
        if return_precision_n_recall:
            return precision, recall
        
        average_precision = np.sum(precision[..., ::-1] * (recall[..., ::-1] - np.concatenate([np.zeros([self.n_class, 1]), recall[..., 1:][..., ::-1]], axis = -1)), axis = -1)
        if reduce:
            average_precision = np.mean(average_precision)
        return average_precision