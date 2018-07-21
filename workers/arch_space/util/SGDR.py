import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingRestartsLR(_LRScheduler):
    def __init__(self, optimizer, num_batches, T_max, T_mul=1, eta_min=0, last_epoch=-1):
        '''
        Here last_epoch actually means last_step since the
        learning rate is decayed after each batch step.
        '''

        self.T_max = T_max
        self.T_mul = T_mul
        self.eta_min = eta_min
        self.num_batches = num_batches
        super(CosineAnnealingRestartsLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        '''
        Override this method to the existing get_lr() of the parent class
        '''
        if self.last_epoch == self.num_batches * self.T_max:
            self.T_max = self.T_max * self.T_mul
            self.last_epoch = 0
        return [self.eta_min + (base_lr - self.eta_min) *
                        (1 + math.cos(math.pi * self.last_epoch / (self.T_max * self.num_batches))) / 2
                        for base_lr in self.base_lrs]
