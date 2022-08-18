import tensorflow as tf
import numpy as np

class WarmUpLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self, epoch = 5, initial_epoch = 0, verbose = 0):
        super(WarmUpLearningRateScheduler, self).__init__(schedule = self.schedule, verbose = verbose)
        self.epoch = epoch
        self.learning_rate = 1.
        self.initial_epoch = initial_epoch
        
    def on_train_begin(self, logs = None):
        self.learning_rate = tf.keras.backend.get_value(self.model.optimizer.lr)
        
    def on_train_end(self, logs = None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.learning_rate)
        
    def schedule(self, epoch, lr):
        epoch += self.initial_epoch
        w = ((epoch + 1) / self.epoch) if epoch < self.epoch else 1
        return self.learning_rate * w

class LinearLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self, cycle, decay_rate = 1e-2, initial_epoch = 0, verbose = 0):
        super(LinearLearningRateScheduler, self).__init__(schedule = self.schedule, verbose = verbose)
        self.cycle = cycle
        self.decay_rate = decay_rate
        self.learning_rate = 1.
        self.initial_epoch = initial_epoch
        
    def on_train_begin(self, logs = None):
        self.learning_rate = tf.keras.backend.get_value(self.model.optimizer.lr)
        
    def on_train_end(self, logs = None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.learning_rate)
        
    def schedule(self, epoch, lr):
        epoch += self.initial_epoch
        w = (1 - (epoch % self.cycle) / (self.cycle - 1)) * (1. - self.learning_rate * self.decay_rate) + self.learning_rate * self.decay_rate
        return self.learning_rate * w
    
class CosineLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self, cycle, decay_rate = 1e-2, initial_epoch = 0, verbose = 0):
        super(CosineLearningRateScheduler, self).__init__(schedule = self.schedule, verbose = verbose)
        self.cycle = cycle
        self.decay_rate = decay_rate
        self.learning_rate = 1.
        self.initial_epoch = initial_epoch
        
    def on_train_begin(self, logs = None):
        self.learning_rate = tf.keras.backend.get_value(self.model.optimizer.lr)
        
    def on_train_end(self, logs = None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.learning_rate)
        
    def schedule(self, epoch, lr):
        epoch += self.initial_epoch
        w = self.decay_rate + (1 - self.decay_rate) * (0.5 * (1 + np.cos(np.pi * (epoch % self.cycle) / self.cycle)))
        return self.learning_rate * w
    
class WarmUpLinearLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self, cycle, decay_rate = 1e-2, warm_up_epoch = 5, initial_epoch = 0, verbose = 0):
        super(WarmUpLinearLearningRateScheduler, self).__init__(schedule = self.schedule, verbose = verbose)
        self.cycle = cycle
        self.decay_rate = decay_rate
        self.learning_rate = 1.
        self.warm_up_epoch = warm_up_epoch
        self.initial_epoch = initial_epoch
        
    def on_train_begin(self, logs = None):
        self.learning_rate = tf.keras.backend.get_value(self.model.optimizer.lr)
        
    def on_train_end(self, logs = None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.learning_rate)
        
    def schedule(self, epoch, lr):
        epoch += self.initial_epoch
        if epoch < self.warm_up_epoch:
            w = ((epoch + 1) / self.warm_up_epoch)
            return self.learning_rate * w
        else:
            w = (1 - ((epoch - self.warm_up_epoch + 1) % self.cycle) / (self.cycle - 1)) * (1. - self.learning_rate * self.decay_rate) + self.learning_rate * self.decay_rate
            return self.learning_rate * w
    
class WarmUpCosineLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self, cycle, decay_rate = 1e-2, warm_up_epoch = 5, initial_epoch = 0, verbose = 0):
        super(WarmUpCosineLearningRateScheduler, self).__init__(schedule = self.schedule, verbose = verbose)
        self.cycle = cycle
        self.decay_rate = decay_rate
        self.learning_rate = 1.
        self.warm_up_epoch = warm_up_epoch
        self.initial_epoch = initial_epoch
        
    def on_train_begin(self, logs = None):
        self.learning_rate = tf.keras.backend.get_value(self.model.optimizer.lr)
        
    def on_train_end(self, logs = None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.learning_rate)
        
    def schedule(self, epoch, lr):
        epoch += self.initial_epoch
        if epoch < self.warm_up_epoch:
            w = ((epoch + 1) / self.warm_up_epoch)
            return self.learning_rate * w
        else:
            w = self.decay_rate + (1 - self.decay_rate) * (0.5 * (1 + np.cos(np.pi * ((epoch - self.warm_up_epoch + 1) % self.cycle) / self.cycle)))
            return self.learning_rate * w