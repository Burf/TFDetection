import tensorflow as tf
import numpy as np

from tfdet.core.util import EMA as ema_util

class EMA(tf.keras.callbacks.Callback):
    """
    1) ema = EMA(decay = 0.9999, auto_apply = False) #auto_apply > ema.apply() in on_epoch_end, ema.restore() in on_epoch_begin
    2) model.fit(...,
                 callbacks=[...,
                            ema,
                            ...])
    
    if step is None, update by epoch. else update by N step
    recommend step > max(round(64 / batch_size), 1)
    """
    def __init__(self, decay = 0.9999, auto_apply = True, step = None, warm_up_epoch = 0, n_update = 0, ramp = 2000, apply_model = None, **kwargs):
        super(EMA, self).__init__(**kwargs)
        self.decay = decay
        self.auto_apply = auto_apply
        self.step = step
        self.warm_up_epoch = warm_up_epoch
        self.n_update = n_update
        self.ramp = ramp
        self.apply_model = apply_model
        
        self.ema = None
        self.model = None
        self._step = step
        self.step_count = 0
        
    def on_train_begin(self, logs = None):
        self.ema = ema_util(self.model, decay = self.decay, n_update = self.n_update, ramp = self.ramp)
    
    def on_epoch_begin(self, epoch, logs = None):
        if self.step is not None:
            if self.warm_up_epoch <= epoch:
                self._step = self.step
            else:
                #https://github.com/WongKinYiu/yolov7
                self._step = max(1, round(np.interp(epoch, [0, self.warm_up_epoch - 1], [1, self.step])))
        if self.auto_apply:
            self.restore(self.apply_model if isinstance(self.apply_model, tf.keras.Model) else self.model)
        
    def on_epoch_end(self, epoch, logs = None):
        if self._step is None:
            self.ema.update()
        if self.auto_apply:
            self.apply(self.apply_model if isinstance(self.apply_model, tf.keras.Model) else self.model)

    def on_train_batch_begin(self, step, logs = None):
        self.step_count += 1
        
    def on_train_batch_end(self, step, logs = None):
        if self._step is not None and round(self.step_count % self._step) == 0:
            self.ema.update()

    def apply(self, model = None):
        if self.ema is not None:
            self.ema.apply(model)
            
    def restore(self, model = None):
        if self.ema is not None:
            self.ema.restore(model)