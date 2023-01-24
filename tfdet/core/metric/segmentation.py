import numpy as np

from tfdet.core.util import metric2text

class MeanIoU:
    def __init__(self, beta = 1, e = 1e-12, label = None, dtype = np.float32):
        """
        run = MeanIoU()(*args)
        batch run = self.add(*batch_args) -> self.evaluate()
        """
        self.beta = beta
        self.e = e
        self.label = label
        self.dtype = dtype
        
        self.reset()
        
    def reset(self):
        self.area_inter = None
        self.area_union = None
        self.area_true = None
        self.area_pred = None
        
    @property
    def iou(self):
        return self.evaluate(reduce = False)
    
    @property
    def mean_iou(self):
        return self.evaluate()
    
    @property
    def accuracy(self):
        return (self.area_inter / np.maximum(self.area_true, self.e)) if self.area_true is not None else [0.]
    
    @property
    def mean_accuracy(self):
        return np.mean(self.accuracy).item()
    
    @property
    def dice(self):
        return (2 * self.area_inter / np.maximum(self.area_true + self.area_pred, self.e)) if self.area_true is not None else [0.]
    
    @property
    def mean_dice(self):
        return np.mean(self.dice).item()
    
    @property
    def precision(self):
        return self.area_inter / np.maximum(self.area_pred, self.e) if self.area_true is not None else [0.]
    
    @property
    def recall(self):
        return self.accuracy
    
    @property
    def f1(self):
        return ((1 + self.beta**2) * np.multiply(self.precision, self.recall)) / np.maximum(np.add(np.multiply(self.beta**2, self.precision), self.recall), self.e)
    
    @property
    def mean_f1(self):
        return np.mean(self.f1).item()
    
    @property
    def summary(self):
        return [self.accuracy, self.iou, self.dice, self.f1]
    
    @property
    def summary_text(self):
        text = ""
        if self.area_true is not None:
            try:
                accuracy, iou, dice, f1 = self.summary
                info = {"accuracy":accuracy, "iou":iou, "dice":dice, "f1":f1}
                
                summary = [np.mean(accuracy).item(), np.mean(iou).item(), np.mean(dice).item(), np.mean(f1).item()]
                
                text = metric2text(info, summary = summary, label = self.label)
            except:
                pass
        return text
    
    def __call__(self, mask_true, mask_pred, reset = True):
        """
        mask_true = #(batch_size, h, w, 1 or n_class)
        mask_pred = #(batch_size, h, w, n_class)
        """
        if reset:
            self.reset()
        self.add(mask_true, mask_pred)
        return self.evaluate()
    
    def add(self, mask_true, mask_pred):
        """
        mask_true = #(h, w, 1 or n_class) or (batch_size, h, w, 1 or n_class)
        mask_pred = #(h, w, n_class) or (batch_size, h, w, n_class)
        """
        if np.ndim(mask_true) == 4:
            for index in range(len(mask_true)):
                self.add(mask_true[index], mask_pred[index])
        else:
            n_class = np.shape(mask_pred)[-1]
            if self.area_true is None:
                self.area_inter = np.zeros(n_class, dtype = self.dtype)
                self.area_union = np.zeros(n_class, dtype = self.dtype)
                self.area_true = np.zeros(n_class, dtype = self.dtype)
                self.area_pred = np.zeros(n_class, dtype = self.dtype)

            if 1 < np.shape(mask_true)[-1]:
                mask_true = np.expand_dims(np.argmax(mask_true, axis = -1), axis = -1)
            mask_pred = np.expand_dims(np.argmax(mask_pred, axis = -1), axis = -1)
            
            inter = mask_pred[mask_pred == mask_true]
            area_inter = np.histogram(inter, bins = n_class, range = (0, n_class - 1))[0]
            area_true = np.histogram(mask_true, bins = n_class, range = (0, n_class - 1))[0]
            area_pred = np.histogram(mask_pred, bins = n_class, range = (0, n_class - 1))[0]
            if self.dtype is not None:
                area_inter = area_inter.astype(self.dtype)
                area_true = area_true.astype(self.dtype)
                area_pred = area_pred.astype(self.dtype)
            area_union = area_true + area_pred - area_inter
            self.area_inter += area_inter
            self.area_union += area_union
            self.area_true += area_true
            self.area_pred += area_pred
        
    def evaluate(self, reduce = True):
        if self.area_true is not None:
            iou = self.area_inter / np.maximum(self.area_union, self.e)
            if reduce:
                iou = np.mean(iou).item()
            return iou
        else:
            return 0. if reduce else [0.]
