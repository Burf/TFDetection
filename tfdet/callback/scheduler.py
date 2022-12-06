import tensorflow as tf
import numpy as np

class LearningRateScheduler(tf.keras.callbacks.Callback):
    """
    schedule = lambda current_epoch, init_learning_rate, current_learning_rate: new_learning_rate
    current_epoch = initial_epoch + current_epoch
    """
    def __init__(self, schedule, initial_epoch = 0, name = "learning_rate"):
        super(LearningRateScheduler, self)
        self.schedule = schedule
        self.initial_epoch = initial_epoch
        self.name = name
        
        self.learning_rate = 1.
        
    def on_train_begin(self, logs = None):
        self.learning_rate = tf.keras.backend.get_value(self.model.optimizer.lr)
        
    def on_train_end(self, logs = None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.learning_rate)
        
    def on_epoch_begin(self, epoch, logs = None):
        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        new_lr = self.schedule(self.initial_epoch + epoch, self.learning_rate, lr)
        if lr != new_lr:
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        
    def on_epoch_end(self, epoch, logs = None):
        if logs is not None:
            logs[self.name] = tf.keras.backend.get_value(self.model.optimizer.lr)
            
class LearningRateSchedulerStep(tf.keras.callbacks.Callback):
    """
    schedule = lambda current_epoch, current_step, total_step, init_learning_rate, current_learning_rate: new_learning_rate
    current_epoch = initial_epoch + current_epoch
    total_step = None if total_step is None else total_step (in first epoch)
    """
    def __init__(self, schedule, total_step = None, initial_epoch = 0, name = "learning_rate"):
        super(LearningRateSchedulerStep, self)
        self.schedule = schedule
        self.total_step = total_step
        self.initial_epoch = initial_epoch
        self.name = name
        
        self.learning_rate = 1.
        self._lr = 1.
        self._epoch = 0
        self.step_count = 0
        
    def on_train_begin(self, logs = None):
        self.learning_rate = tf.keras.backend.get_value(self.model.optimizer.lr)
        
    def on_train_end(self, logs = None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.learning_rate)
        
    def on_epoch_begin(self, epoch, logs = None):
        self._lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        self._epoch = epoch
        
    def on_epoch_end(self, epoch, logs = None):
        if self.total_step is None:
            self.total_step = self.step_count
        if logs is not None:
            logs[self.name] = tf.keras.backend.get_value(self.model.optimizer.lr)
        
    def on_train_batch_begin(self, step, logs = None):
        if self.total_step is None:
            self.step_count += 1
        new_lr = self.schedule(self.initial_epoch + self._epoch, step, self.total_step, self.learning_rate, self._lr)
        if self._lr != new_lr:
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            self._lr = new_lr
            
    #def on_train_batch_end(self, step, logs = None):
    #    if logs is not None:
    #        logs[self.name] = self._lr

class WarmUpLearningRateScheduler(LearningRateScheduler):
    def __init__(self, epoch = 5, initial_epoch = 0, name = "learning_rate"):
        super(WarmUpLearningRateScheduler, self).__init__(schedule = self.schedule, initial_epoch = initial_epoch, name = name)
        self.epoch = epoch
        
    def schedule(self, epoch, learning_rate, current_learning_rate):
        if epoch < self.epoch:
            w = ((epoch + 1) / self.epoch)
        else:
            w = 1
        return learning_rate * w

class LinearLearningRateScheduler(LearningRateScheduler):
    def __init__(self, cycle, decay_rate = 1e-2, initial_epoch = 0, name = "learning_rate"):
        super(LinearLearningRateScheduler, self).__init__(schedule = self.schedule, initial_epoch = initial_epoch, name = name)
        self.cycle = cycle
        self.decay_rate = decay_rate
        
    def schedule(self, epoch, learning_rate, current_learning_rate):
        w = (1 - (epoch % self.cycle) / (self.cycle - 1)) * (1. - learning_rate * self.decay_rate) + learning_rate * self.decay_rate
        return learning_rate * w
    
class CosineLearningRateScheduler(LearningRateScheduler):
    def __init__(self, cycle, decay_rate = 1e-2, initial_epoch = 0, name = "learning_rate"):
        super(CosineLearningRateScheduler, self).__init__(schedule = self.schedule, initial_epoch = initial_epoch, name = name)
        self.cycle = cycle
        self.decay_rate = decay_rate
        
    def schedule(self, epoch, learning_rate, current_learning_rate):
        w = self.decay_rate + (1 - self.decay_rate) * (0.5 * (1 + np.cos(np.pi * (epoch % self.cycle) / self.cycle)))
        return learning_rate * w
    
class WarmUpLinearLearningRateScheduler(LearningRateScheduler):
    def __init__(self, cycle, decay_rate = 1e-2, warm_up_epoch = 5, initial_epoch = 0, name = "learning_rate"):
        super(WarmUpLinearLearningRateScheduler, self).__init__(schedule = self.schedule, initial_epoch = initial_epoch, name = name)
        self.cycle = cycle
        self.decay_rate = decay_rate
        self.warm_up_epoch = warm_up_epoch
        
    def schedule(self, epoch, learning_rate, current_learning_rate):
        if epoch < self.warm_up_epoch:
            w = ((epoch + 1) / self.warm_up_epoch)
        else:
            w = (1 - (((epoch + 1) - self.warm_up_epoch) % self.cycle) / (self.cycle - 1)) * (1. - learning_rate * self.decay_rate) + learning_rate * self.decay_rate
        return learning_rate * w
    
class WarmUpCosineLearningRateScheduler(LearningRateScheduler):
    def __init__(self, cycle, decay_rate = 1e-2, warm_up_epoch = 5, initial_epoch = 0, name = "learning_rate"):
        super(WarmUpCosineLearningRateScheduler, self).__init__(schedule = self.schedule, initial_epoch = initial_epoch, name = name)
        self.cycle = cycle
        self.decay_rate = decay_rate
        self.warm_up_epoch = warm_up_epoch
        
    def schedule(self, epoch, learning_rate, current_learning_rate):
        if epoch < self.warm_up_epoch:
            w = ((epoch + 1) / self.warm_up_epoch)
        else:
            w = self.decay_rate + (1 - self.decay_rate) * (0.5 * (1 + np.cos(np.pi * (((epoch + 1) - self.warm_up_epoch) % self.cycle) / self.cycle)))
        return learning_rate * w
    
class WarmUpLearningRateSchedulerStep(LearningRateSchedulerStep):
    def __init__(self, epoch = 5, total_step = None, initial_epoch = 0, name = "learning_rate"):
        super(WarmUpLearningRateSchedulerStep, self).__init__(schedule = self.schedule, total_step = total_step, initial_epoch = initial_epoch, name = name)
        self.epoch = epoch
        
    def schedule(self, epoch, step, total_step, learning_rate, current_learning_rate):
        total_step = 1000 if total_step is None else total_step
        if epoch < self.epoch:
            w = np.interp(step + 1, [0, total_step], [epoch / self.epoch, (epoch + 1) / self.epoch])
        else:
            w = 1
        return learning_rate * w

class LinearLearningRateSchedulerStep(LearningRateSchedulerStep):
    def __init__(self, cycle, decay_rate = 1e-2, total_step = None, initial_epoch = 0, name = "learning_rate"):
        super(LinearLearningRateSchedulerStep, self).__init__(schedule = self.schedule, total_step = total_step, initial_epoch = initial_epoch, name = name)
        self.cycle = cycle
        self.decay_rate = decay_rate
        
    def schedule(self, epoch, step, total_step, learning_rate, current_learning_rate):
        total_step = 1000 if total_step is None else total_step
        w = (1 - (epoch % self.cycle) / (self.cycle - 1)) * (1. - learning_rate * self.decay_rate) + learning_rate * self.decay_rate
        w2 = (1 - ((epoch + 1) % self.cycle) / (self.cycle - 1)) * (1. - learning_rate * self.decay_rate) + learning_rate * self.decay_rate
        w = np.interp(step, [0, total_step], [w, w2])
        return learning_rate * w
    
class CosineLearningRateSchedulerStep(LearningRateSchedulerStep):
    def __init__(self, cycle, decay_rate = 1e-2, total_step = None, initial_epoch = 0, name = "learning_rate"):
        super(CosineLearningRateSchedulerStep, self).__init__(schedule = self.schedule, total_step = total_step, initial_epoch = initial_epoch, name = name)
        self.cycle = cycle
        self.decay_rate = decay_rate
        
    def schedule(self, epoch, step, total_step, learning_rate, current_learning_rate):
        total_step = 1000 if total_step is None else total_step
        w = self.decay_rate + (1 - self.decay_rate) * (0.5 * (1 + np.cos(np.pi * (epoch % self.cycle) / self.cycle)))
        w2 = self.decay_rate + (1 - self.decay_rate) * (0.5 * (1 + np.cos(np.pi * ((epoch + 1) % self.cycle) / self.cycle)))
        w = np.interp(step, [0, total_step], [w, w2])
        return learning_rate * w
    
class WarmUpLinearLearningRateSchedulerStep(LearningRateSchedulerStep):
    def __init__(self, cycle, decay_rate = 1e-2, warm_up_epoch = 5, total_step = None, initial_epoch = 0, name = "learning_rate"):
        super(WarmUpLinearLearningRateSchedulerStep, self).__init__(schedule = self.schedule, total_step = total_step, initial_epoch = initial_epoch, name = name)
        self.cycle = cycle
        self.decay_rate = decay_rate
        self.warm_up_epoch = warm_up_epoch
        
    def schedule(self, epoch, step, total_step, learning_rate, current_learning_rate):
        total_step = 1000 if total_step is None else total_step
        if epoch < self.warm_up_epoch:
            w = np.interp(step + 1, [0, total_step], [epoch / self.warm_up_epoch, (epoch + 1) / self.warm_up_epoch])
        else:
            w = (1 - ((epoch - self.warm_up_epoch) % self.cycle) / (self.cycle - 1)) * (1. - learning_rate * self.decay_rate) + learning_rate * self.decay_rate
            w2 = (1 - (((epoch + 1) - self.warm_up_epoch) % self.cycle) / (self.cycle - 1)) * (1. - learning_rate * self.decay_rate) + learning_rate * self.decay_rate
            w = np.interp(step, [0, total_step], [w, w2])
        return learning_rate * w
    
class WarmUpCosineLearningRateSchedulerStep(LearningRateSchedulerStep):
    def __init__(self, cycle, decay_rate = 1e-2, warm_up_epoch = 5, total_step = None, initial_epoch = 0, name = "learning_rate"):
        super(WarmUpCosineLearningRateSchedulerStep, self).__init__(schedule = self.schedule, total_step = total_step, initial_epoch = initial_epoch, name = name)
        self.cycle = cycle
        self.decay_rate = decay_rate
        self.warm_up_epoch = warm_up_epoch
        
    def schedule(self, epoch, step, total_step, learning_rate, current_learning_rate):
        total_step = 1000 if total_step is None else total_step
        if epoch < self.warm_up_epoch:
            w = np.interp(step + 1, [0, total_step], [epoch / self.warm_up_epoch, (epoch + 1) / self.warm_up_epoch])
        else:
            w = self.decay_rate + (1 - self.decay_rate) * (0.5 * (1 + np.cos(np.pi * ((epoch - self.warm_up_epoch) % self.cycle) / self.cycle)))
            w2 = self.decay_rate + (1 - self.decay_rate) * (0.5 * (1 + np.cos(np.pi * (((epoch + 1) - self.warm_up_epoch) % self.cycle) / self.cycle)))
            w = np.interp(step, [0, total_step], [w, w2])
        return learning_rate * w
