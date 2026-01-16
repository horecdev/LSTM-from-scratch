import cupy as cp
import numpy as np

class CosineScheduler:
    def __init__(self, max_lr, min_lr, warmup_epochs, total_epochs):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def get_lr(self, epoch): # assumes that starts from 1
        if epoch <= self.warmup_epochs:
            return self.max_lr * (epoch / self.warmup_epochs)
        
        else:
            decay_steps = self.total_epochs - self.warmup_epochs
            current_step = epoch - self.warmup_epochs
            ratio = current_step / decay_steps
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * ratio))