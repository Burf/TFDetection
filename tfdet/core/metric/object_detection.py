import tensorflow as tf
import numpy as np

from tfdet.core.util import metric2text
from ..bbox import overlap_bbox_numpy as overlap_bbox

class MeanAveragePrecision:
    def __init__(self, iou_threshold = 0.5, score_threshold = 0.05, scale_range = None, mode = "normal", e = 1e-12, postfix = False, label = None, dtype = np.float32):
        """
        run = MeanAveragePrecision()(*args)
        batch run = self.add(*batch_args) -> self.evaluate()
        
        scale_range = None > area_range = [[None, None]] # 0~INF (all scale)
        scale_range = [96] > area_range = [[None, 96^2], [96^2, None]] # 0~96^2, 96^2~INF
        scale_range = [32, 96] > area_range = [[None, 32^2], [32^2, 96^2], [96^2, None]] # 0~32^2, 32^2~96^2, 96^2~INF
        scale_range = [None, 32, 96] > area_range = [[None, None], [None, 32^2], [32^2, 96^2], [96^2, None]] #0~INF, 0~32^2, 32^2~96^2, 96^2~INF
        scale_range = [32, None, 96] > area_range = [[None, 32^2], [32^2, None], [None, 96^2], [96^2, None]] #0~32^2, 32^2~INF, 0~96^2, 96^2~INF
        """
        self.iou_threshold = np.array([iou_threshold]) if np.ndim(iou_threshold) == 0 else np.array(iou_threshold)
        self.score_threshold = score_threshold
        self.mode = mode
        self.scale_range = scale_range
        self.e = e
        self.postfix = postfix
        self.label = label
        self.dtype = dtype

        if np.ndim(scale_range) == 0:
            scale_range = [scale_range]
        if np.all([scale is None for scale in scale_range]):
            scale_range = [None, None]
        else:
            sorted_scale_range = sorted([scale for scale in scale_range if scale is not None])
            scale_range = [sorted_scale_range.pop(0) if scale is not None else None for scale in scale_range]
            scale_range = [None, *scale_range, None]
        self.area_range = [[scale_range[i]**2 if scale_range[i] is not None else None, scale_range[i + 1]**2  if scale_range[i + 1] is not None else None] for i in range(len(scale_range) - 1)]
        self.reset()
        
    def reset(self):
        self.tp = []
        self.fp = []
        self.score_pred = []
        self._num_true = None
        self._num_pred = None
    
    @property
    def num_true(self):
        if self._num_true is not None:
            return self._num_true[0] if self.scale_range is None else self._num_true
    
    @property
    def num_pred(self):
        if self._num_pred is not None:
            return self._num_pred[0] if self.scale_range is None else self._num_pred
    
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
        precision, recall = self.evaluate(reduce = False, return_precision_n_recall = True)
        f1 = 2 * (precision * recall) / np.maximum(precision + recall, self.e)
        return f1
    
    @property
    def average_precision(self):
        return self.evaluate(reduce = False)
    
    @property
    def mean_average_precision(self):
        return self.evaluate()
    
    @property
    def summary(self):
        return self.evaluate(reduce = False, return_summary = True)
    
    @property
    def summary_text(self):
        text = ""
        if self._num_pred is not None and 0 < np.sum(self._num_pred):
            try:
                precision, recall, average_precision = self.summary

                num_true = self._num_true[..., 0].astype(int)
                num_pred = self._num_pred[..., 0].astype(int)
                mean_recall, mean_average_precision = self.reduce([recall, average_precision])
                recall, average_precision = np.mean(recall, axis = -1), np.mean(average_precision, axis = -1)
                if self.scale_range is None:
                    recall, average_precision = [recall], [average_precision]
                    mean_recall, mean_average_precision = [mean_recall], [mean_average_precision]
                    
                if self.postfix:
                    min_iou_threshold = str(np.min(self.iou_threshold)).replace("0.", ".")
                    max_iou_threshold = str(np.max(self.iou_threshold)).replace("0.", ".")
                    postfix = "@{0}".format(min_iou_threshold)
                    if min_iou_threshold != max_iou_threshold:
                        postfix = "{0}:{1}".format(postfix, max_iou_threshold)
                else:
                    postfix = ""

                post = lambda x:int(x)
                if self.scale_range is not None:
                    scale_range = [self.scale_range] if np.ndim(self.scale_range) == 0 else self.scale_range
                    scale_range = [scale for scale in scale_range if scale is not None]
                    if np.issubdtype(np.array(scale_range).dtype, np.floating):
                        post = lambda x:float(x)
                area_range = [[str(post(min_area**(1/2) if min_area is not None else 0)), str(post(max_area**(1/2))) if max_area is not None else "INF"] for (min_area, max_area) in self.area_range]
                all_range_flag = [float(min_area) == 0 and max_area == "INF" for (min_area, max_area) in area_range]
                scale_range_flag = np.invert(all_range_flag)
                
                all_range_exist = 0 < np.sum(all_range_flag)
                if all_range_exist:
                    i = np.argwhere(all_range_flag)[:, 0][0]
                    num_true = num_true[i]
                    num_pred = num_pred[i]
                    info = {"num_true":num_true, "num_pred":num_pred,
                            "recall{0}".format(postfix):recall[i], "average_precision{0}".format(postfix):average_precision[i]}
                    summary = [np.sum(num_true), np.sum(num_pred), mean_recall[i], mean_average_precision[i]] 
                else:
                    num_true = np.sum(num_true[scale_range_flag], axis = 0)
                    num_pred = np.sum(num_pred[scale_range_flag], axis = 0)
                    info = {"num_true":num_true, "num_pred":num_pred}
                    summary = [np.sum(num_true), np.sum(num_pred)]
                for i in np.argwhere(scale_range_flag)[:, 0]:
                    if not all_range_exist:
                        info["recall{0}[{1}:{2}]".format(postfix, *area_range[i])] = recall[i]
                        summary.append(mean_recall[i])
                    info["average_precision{0}[{1}:{2}]".format(postfix, *area_range[i])] = average_precision[i]
                    summary.append(mean_average_precision[i])
                text = metric2text(info, summary = summary, label = self.label)
            except:
                pass
        return text
    
    def reduce(self, raw_array, num_true = None):
        """
        (num_area_range, num_class, num_iou_threshold) > (num_area_range)
        (num_class, num_iou_threshold) > (,)
        """
        raw_arrays = [raw_array] if not isinstance(raw_array, (list, tuple)) else raw_array
        if self._num_true is not None:
            align_raw_arrays = [array if np.ndim(array) == 3 else np.expand_dims(array, axis = 0) for array in raw_arrays]
            mean = [[] for _ in range(len(align_raw_arrays))]
            for a in range(len(self.area_range)):
                true_flag = self._num_true[a, ..., 0] != 0
                true_count = np.sum(true_flag)
                for m, array in zip(mean, align_raw_arrays):
                    m.append(np.mean(array[a, true_flag]) if 0 < true_count else 0.)
            result = [np.array(m) if np.ndim(array) == 3 else m[0] for m, array in zip(mean, raw_arrays)]
            if not isinstance(raw_array, (list, tuple)):
                result = result[0]
        else:
            result = np.zeros(len(raw_arrays[0]), dtype = raw_arrays[0].dtype) if np.ndim(raw_arrays[0]) == 3 else 0.
        return result
    
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
            if self._num_true is None:
                self._num_true = np.zeros((len(self.area_range), n_class, 1))
                self._num_pred = np.zeros((len(self.area_range), n_class, 1))
                
            y_true = np.array(y_true, dtype = self.dtype) if not isinstance(y_true, np.ndarray) else (y_true.astype(self.dtype) if self.dtype is not None else y_true)
            bbox_true = np.array(bbox_true, dtype = self.dtype) if not isinstance(bbox_true, np.ndarray) else (bbox_true.astype(self.dtype) if self.dtype is not None else bbox_true)
            y_pred = np.array(y_pred, dtype = self.dtype) if not isinstance(y_pred, np.ndarray) else (y_pred.astype(self.dtype) if self.dtype is not None else y_pred)
            bbox_pred = np.array(bbox_pred, dtype = self.dtype) if not isinstance(bbox_pred, np.ndarray) else (bbox_pred.astype(self.dtype) if self.dtype is not None else bbox_pred)
            
            if 1 < np.shape(y_true)[-1]:
                y_true = np.expand_dims(np.argmax(y_true, axis = -1), axis = -1)
            if np.shape(y_pred)[-1] == 1:
                y_true = (y_true != 0).astype(y_true.dtype)
            #valid_indices = np.where(np.max(0 < bbox_true, axis = -1))
            valid_indices = np.where(np.any(0 < bbox_true, axis = -1))[0]
            y_true = y_true[valid_indices]
            bbox_true = bbox_true[valid_indices]
            #valid_indices = np.where(np.max(0 < bbox_pred, axis = -1))
            valid_indices = np.where(np.any(0 < bbox_pred, axis = -1))[0]
            y_pred = y_pred[valid_indices]
            bbox_pred = bbox_pred[valid_indices]
            
            label = np.argmax(y_pred, axis = -1)
            score = np.max(y_pred, axis = -1)
            sort_indices = np.argsort(-score)
            label = label[sort_indices]
            score = score[sort_indices]
            bbox_pred = bbox_pred[sort_indices]
            
            score_flag = self.score_threshold <= score
            true_area = (bbox_true[..., 3] - bbox_true[..., 1]) * (bbox_true[..., 2] - bbox_true[..., 0])
            pred_area = (bbox_pred[..., 3] - bbox_pred[..., 1]) * (bbox_pred[..., 2] - bbox_pred[..., 0])
            
            tp = np.zeros((len(self.area_range), n_class, len(y_pred), len(self.iou_threshold)))
            fp = np.zeros((len(self.area_range), n_class, len(y_pred), len(self.iou_threshold)))
            for cls in range(n_class):
                true_flag = y_true[..., 0] == cls
                pred_flag = np.logical_and(label == cls, score_flag)
                true_indices = np.argwhere(true_flag)[:, 0]
                pred_indices = np.argwhere(pred_flag)[:, 0]
                cls_true_area = true_area[true_flag]
                cls_pred_area = pred_area[pred_flag]
                if 0 < len(pred_indices) and 0 < len(true_indices):
                    overlaps = overlap_bbox(bbox_pred[pred_flag], bbox_true[true_flag], mode = self.mode) #(P, T)
                    max_iou = np.max(overlaps, axis = 1) #(P,)
                    argmax_iou = np.argmax(overlaps, axis = 1) #(P,)
                for a, (min_area, max_area) in enumerate(self.area_range):
                    if 0 < len(true_indices):
                        true_area_flag = np.ones(len(true_indices), dtype = bool) #(T,)
                        if min_area is not None or max_area is not None:
                            min_area_flag = np.greater_equal(cls_true_area, min_area) if min_area is not None else true_area_flag
                            max_area_flag = np.less(cls_true_area, max_area) if max_area is not None else true_area_flag
                            true_area_flag = np.logical_and(min_area_flag, max_area_flag)
                        self._num_true[a, cls] += np.sum(true_area_flag)
                    if 0 < len(pred_indices):
                        pred_area_flag = np.ones(len(cls_pred_area), dtype = bool)
                        if min_area is not None or max_area is not None:
                            min_area_flag = np.greater_equal(cls_pred_area, min_area) if min_area is not None else pred_area_flag
                            max_area_flag = np.less(cls_pred_area, max_area) if max_area is not None else pred_area_flag
                            pred_area_flag = np.logical_and(min_area_flag, max_area_flag)
                        self._num_pred[a, cls] += np.sum(pred_area_flag)
                        if len(true_indices) == 0:
                            fp[a, cls, pred_indices[pred_area_flag]] = 1
                        else:
                            for i, iou_threshold in enumerate(self.iou_threshold):
                                iou_flag = iou_threshold <= max_iou #(P,)
                                match_true_indices = argmax_iou[iou_flag]
                                match_pred_indices = np.argwhere(iou_flag)[:, 0]
                                filtered_flag = true_area_flag[match_true_indices]
                                match_true_indices = match_true_indices[filtered_flag]
                                match_pred_indices = match_pred_indices[filtered_flag]
                                unmatch_pred_indices = np.argwhere(np.logical_and(~iou_flag, pred_area_flag))[:, 0]
                                #match update
                                for t in np.unique(match_true_indices):
                                    p = match_pred_indices[match_true_indices == t]
                                    tp[a, cls, p[0], i] = 1
                                    fp[a, cls, p[1:], i] = 1
                                #unmatch update
                                fp[a, cls, unmatch_pred_indices, i] = 1
            self.tp.append(tp)
            self.fp.append(fp)
            self.score_pred.append(score)
    
    def evaluate(self, reduce = True, return_precision_n_recall = False, mode = "area", return_summary = False):
        if self._num_pred is not None and 0 < np.sum(self._num_pred): #self._num_true is not None:
            if mode not in ["area", "11points"]:
                raise ValueError("unknown mode '{0}'".format(mode))

            sort_indices = np.argsort(-np.hstack(self.score_pred))
            tp = np.concatenate(self.tp, axis = -2)[..., sort_indices, :]
            fp = np.concatenate(self.fp, axis = -2)[..., sort_indices, :]
            tp = np.cumsum(tp, axis = -2)
            fp = np.cumsum(fp, axis = -2)
            num_true = np.expand_dims(self._num_true, axis = -2)
            true_flag = num_true != 0
            precision = tp / np.maximum(tp + fp, self.e)
            recall = tp / np.maximum(num_true, self.e)
            precision = np.where(true_flag, precision, 0)
            recall = np.where(true_flag, recall, 0)
            
            n_class = np.shape(self.tp[0])[-3]
            average_precision = np.zeros((len(self.area_range), n_class, len(self.iou_threshold)))
            if not return_precision_n_recall:
                if mode == "area":
                    zeros = np.zeros((len(self.area_range), n_class, 1, len(self.iou_threshold)))
                    ones = np.ones((len(self.area_range), n_class, 1, len(self.iou_threshold)))
                    mpre = np.concatenate([zeros, precision, zeros], axis = -2)
                    mrec = np.concatenate([zeros, recall, ones], axis = -2)
                    for i in range(np.shape(mpre)[-2] - 1, 0, -1):
                        mpre[..., i - 1, :] = np.maximum(mpre[..., i - 1, :], mpre[..., i, :])
                    
                    for a in range(len(self.area_range)):
                        for cls in range(n_class):
                            for th in range(len(self.iou_threshold)):
                                indices = np.where(mrec[a, cls, 1:, th] != mrec[a, cls, :-1, th])[0]
                                average_precision[a, cls, th] = np.sum((mrec[a, cls, indices + 1, th] - mrec[a, cls, indices, th]) * mpre[a, cls, indices + 1, th])
                elif mode in "11points":
                    zeros = np.zeros_like(precision)
                    for recall_threshold in np.linspace(0., 1., 11):
                        pre = np.where(recall_threshold <= recall, precision, zeros)
                        average_precision += np.max(pre, axis = -2)
                    average_precision /= 11
            precision = precision[..., -1, :]
            recall = recall[..., -1, :]
            
            if reduce:
                precision, recall, average_precision = self.reduce([precision, recall, average_precision])
        else:
            n_class = 0
            if reduce:
                precision = recall = average_precision = np.zeros(len(self.area_range))
            else:
                precision = recall = average_precision = np.zeros((len(self.area_range), n_class, len(self.iou_threshold)))
            
        if self.scale_range is None:
            precision, recall, average_precision = precision[0], recall[0], average_precision[0]

        if return_precision_n_recall:
            return [precision, recall]
        elif return_summary:
            return [precision, recall, average_precision]
        else:
            return average_precision

class CoCoMeanAveragePrecision:
    def __init__(self, iou_threshold = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], score_threshold = 0.05, scale_range = None, mode = "normal", e = 1e-12, postfix = False, label = None, dtype = np.float32):
        """
        run = CoCoMeanAveragePrecision()(*args)
        batch run = self.add(*batch_args) -> self.evaluate()
        
        scale_range = None > area_range = [[None, None]] # 0~INF (all scale)
        scale_range = [96] > area_range = [[None, 96^2], [96^2, None]] # 0~96^2, 96^2~INF
        scale_range = [32, 96] > area_range = [[None, 32^2], [32^2, 96^2], [96^2, None]] # 0~32^2, 32^2~96^2, 96^2~INF
        scale_range = [None, 32, 96] > area_range = [[None, None], [None, 32^2], [32^2, 96^2], [96^2, None]] #0~INF, 0~32^2, 32^2~96^2, 96^2~INF
        scale_range = [32, None, 96] > area_range = [[None, 32^2], [32^2, None], [None, 96^2], [96^2, None]] #0~32^2, 32^2~INF, 0~96^2, 96^2~INF
        """
        self.iou_threshold = np.array([iou_threshold]) if np.ndim(iou_threshold) == 0 else np.array(iou_threshold)
        self.score_threshold = score_threshold
        self.scale_range = scale_range
        self.mode = mode
        self.e = e
        self.postfix = postfix
        self.label = label
        self.dtype = dtype
        
        self.reset()
        
    def reset(self):
        self.metric = MeanAveragePrecision(iou_threshold = self.iou_threshold, score_threshold = self.score_threshold, scale_range = self.scale_range, mode = self.mode, e = self.e, dtype = self.dtype)
        self.sub_metric = MeanAveragePrecision(iou_threshold = [0.5, 0.75], score_threshold = self.score_threshold, scale_range = None, mode = self.mode, e = self.e, dtype = self.dtype)
    
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
    def summary(self):
        return self.metric.summary
    
    @property
    def num_true_50(self):
        return self.sub_metric.num_true
    
    @property
    def num_pred_50(self):
        return self.sub_metric.num_pred

    @property
    def precision_n_recall_50(self):
        return [array[..., :1] for array in self.sub_metric.precision_n_recall]
    
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
        if self.num_true is not None:
            return np.mean(self.average_precision_50[self.num_true[..., 0] != 0])
        else:
            return 0.
    
    @property
    def summary_50(self):
        return [array[..., :1] for array in self.sub_metric.evaluate(reduce = False, return_summary = True)]
    
    @property
    def num_true_75(self):
        return self.sub_metric.num_true
    
    @property
    def num_pred_75(self):
        return self.sub_metric.num_pred

    @property
    def precision_n_recall_75(self):
        return [array[..., 1:] for array in self.sub_metric.precision_n_recall]
    
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
        if self.num_true is not None:
            return np.mean(self.average_precision_75[self.num_true[..., 0] != 0])
        else:
            return 0.
    
    @property
    def summary_75(self):
        return [array[..., 1:] for array in self.sub_metric.evaluate(reduce = False, return_summary = True)]
    
    @property
    def summary_text(self):
        text = ""
        if self.num_pred is not None and 0 < np.sum(self.num_pred):
            try:
                precision, recall, average_precision = self.metric.summary
                sub_precision, sub_recall, sub_average_precision = self.sub_metric.summary

                num_true = self.metric._num_true[..., 0].astype(int)
                num_pred = self.metric._num_pred[..., 0].astype(int)
                sub_num_true = self.sub_metric.num_true[..., 0].astype(int)
                sub_num_pred = self.sub_metric.num_pred[..., 0].astype(int)
                recall_50, recall_75 = sub_recall[..., :1], sub_recall[..., 1:]
                average_precision_50, average_precision_75 = sub_average_precision[..., :1], sub_average_precision[..., 1:]
                mean_recall, mean_average_precision = self.metric.reduce([recall, average_precision])
                mean_recall_50, mean_average_precision_50, mean_recall_75, mean_average_precision_75 = self.sub_metric.reduce([recall_50, average_precision_50, recall_75, average_precision_75])
                recall, average_precision = np.mean(recall, axis = -1), np.mean(average_precision, axis = -1)
                recall_50, average_precision_50, recall_75, average_precision_75 = recall_50[..., 0], average_precision_50[..., 0], recall_75[..., 0], average_precision_75[..., 0]
                if self.scale_range is None:
                    recall, average_precision = [recall], [average_precision]
                    mean_recall, mean_average_precision = [mean_recall], [mean_average_precision]
                    
                if self.postfix:
                    min_iou_threshold = str(np.min(self.iou_threshold)).replace("0.", ".")
                    max_iou_threshold = str(np.max(self.iou_threshold)).replace("0.", ".")
                    postfix = "@{0}".format(min_iou_threshold)
                    if min_iou_threshold != max_iou_threshold:
                        postfix = "{0}:{1}".format(postfix, max_iou_threshold)
                else:
                    postfix = ""
                
                post = lambda x:int(x)
                if self.scale_range is not None:
                    scale_range = [self.scale_range] if np.ndim(self.scale_range) == 0 else self.scale_range
                    scale_range = [scale for scale in scale_range if scale is not None]
                    if np.issubdtype(np.array(scale_range).dtype, np.floating):
                        post = lambda x:float(x)
                area_range = [[str(post(min_area**(1/2) if min_area is not None else 0)), str(post(max_area**(1/2))) if max_area is not None else "INF"] for (min_area, max_area) in self.metric.area_range]
                all_range_flag = [float(min_area) == 0 and max_area == "INF" for (min_area, max_area) in area_range]
                scale_range_flag = np.invert(all_range_flag)
                
                all_range_exist = 0 < np.sum(all_range_flag)
                if all_range_exist:
                    i = np.argwhere(all_range_flag)[:, 0][0]
                    num_true = num_true[i]
                    num_pred = num_pred[i]
                    info = {"num_true":num_true, "num_pred":num_pred,
                            "recall{0}".format(postfix):recall[i], "average_precision{0}".format(postfix):average_precision[i],
                            "average_precision@.5":average_precision_50,
                            "average_precision@.75":average_precision_75}
                    summary = [np.sum(num_true), np.sum(num_pred), mean_recall[i], mean_average_precision[i], mean_average_precision_50, mean_average_precision_75] 
                else:
                    num_true = np.sum(num_true[scale_range_flag], axis = 0)
                    num_pred = np.sum(num_pred[scale_range_flag], axis = 0)
                    info = {"num_true":num_true, "num_pred":num_pred,
                            "recall@.5":recall_50, "average_precision@.5":average_precision_50,
                            "recall@.75":recall_75, "average_precision@.75":average_precision_75}
                    summary = [np.sum(num_true), np.sum(num_pred), mean_recall_50, mean_average_precision_50, mean_recall_75, mean_average_precision_75]
                for i in np.argwhere(scale_range_flag)[:, 0]:
                    if not all_range_exist:
                        info["recall{0}[{1}:{2}]".format(postfix, *area_range[i])] = recall[i]
                        summary.append(mean_recall[i])
                    info["average_precision{0}[{1}:{2}]".format(postfix, *area_range[i])] = average_precision[i]
                    summary.append(mean_average_precision[i])
                text = metric2text(info, summary = summary, label = self.label)
            except:
                pass
        return text
        
    def reduce(self, raw_array, num_true = None):
        reduce = self.metric.reduce
        if np.ndim(([raw_array] if not isinstance(raw_array, (list, tuple)) else raw_array)[0]) == 2 and self.scale_range is not None:
            reduce = self.sub_metric.reduce
        return reduce(raw_array, num_true)
    
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